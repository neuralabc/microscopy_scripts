"""Application script for running 2D slice registration on specific dataset."""
from slice_registration_functions import *

# file parameters
subject = 'zefir'

# set the resolution of the input images
in_plane_res_x = 10 #10 microns per pixel
in_plane_res_y = 10 #10 microns per pixel
in_plane_res_z = 50 #slice thickness of 50 microns

zfill_num = 4 #number of digits to use for zero-filling in file names, e.g. 4 means that slice the first slice will be 0000
per_slice_template = True #use a median of the slice and adjacent slices to create a slice-specific template for anchoring the registration
use_nonlin_slice_templates = False #use interpolated slices (from registrations of neighbouring 2 slices) as templates for registration, otherwise median
                                    # nonlinear slice templates take a long time and result in very jagged registrations, but may end up being useful for bring slices that are very far out of alignment back in
                                    # currently BROKEN
slice_template_type = 'median'
across_slice_smoothing_sigma = None # (None/0; pos int} sigma for smoothing across the stack (only in the slice direction), applied after stacking and before template creation
if use_nonlin_slice_templates:
    slice_template_type = [slice_template_type,'nonlin']

#this fails on server, for some reason?    
mask_zero = False #mask zeros for nighres registrations

# Control whether to use resolution information during registration
# False (default): Registration works in voxel space (ignore_res=True), better empirical performance
#                  Output images will have the specified voxel resolution in their headers, but registration itself does not use this information
# True: Registration uses physical resolution (ignore_res=False), more physically accurate
use_resolution_in_registration = True

# scaling factor that is applied to the x and y dimensions (in-plane dimensions) to downsample the data
rescale=5 #larger scale means that you have to change the scaling_factor, which is now done automatically just before computations
rescale=40
# rescale=10

#based on the rescale value, we adjust our in-plane resolution
#keep resolutions in microns (not mm) - only apply rescale to x and y
rescaled_in_plane_res_x = rescale * in_plane_res_x
rescaled_in_plane_res_y = rescale * in_plane_res_y
# z resolution stays at original value (not rescaled)
in_plane_res_z_microns = in_plane_res_z

actual_voxel_res = [rescaled_in_plane_res_x, rescaled_in_plane_res_y, in_plane_res_z_microns]
#if we don't want to set the voxel resolution, we can set it to None and it will be 1x1x1
voxel_res = actual_voxel_res # defines voxel resolution for output template in microns # registration itself performs much better when we do not specify the res

downsample_parallel = False #True means that we invoke Parallel, but can be much faster on HPC when set to False since it skips the Parallel overhead
max_workers = 50 #number of parallel workers to run for registration -> registration is slow but not CPU bound on an HPC (192 cores could take ??)
nonlin_interp_max_workers = 50 #number of workers to use for nonlinear slice interpolation when use_nonlin_slice_templates = True

# setup the output directory for use
output_dir = f'/tmp/{subject}_sliceReg_optimized_v2_rescale_{rescale}_sdf/'
_df = pd.read_csv('/data/neuralabc/neuralabc_volunteers/macaque/all_TP_image_idxs_file_lookup.csv')
missing_idxs_to_fill = [32,59,120,160,189,228] #these are the slice indices with missing or terrible data, fill with coreg of neighbours
# output_dir = '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/slice_reg_perSliceTemplate_image_weights_all_tmp/'
## _df = pd.read_csv('/data/data_drive/Macaque_CB/processing/results_from_cell_counts/all_TP_image_idxs_file_lookup.csv')

#missing_idxs_to_fill = [32]
# missing_idxs_to_fill = [5]
# missing_idxs_to_fill = None
all_image_fnames = list(_df['file_name'].values)

# ## for testing XXX
# all_image_fnames = all_image_fnames[0:35] #for testing
# missing_idxs_to_fill = [missing_idxs_to_fill[0]]

print('*********************************************************************************************************')
print(f'Output directory: {output_dir}')
print('*********************************************************************************************************')

# set missing indices, which will be iteratively filled with the mean of the neighbouring slices
if missing_idxs_to_fill is not None:
    if numpy.max(numpy.array(missing_idxs_to_fill)) > len(all_image_fnames): #since these are indices, will start @ 0
        raise ValueError("Missing slice indices exceed the number of images in the stack.")

# all_image_fnames = all_image_fnames[0:10] #for testing
all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below


if not os.path.exists(output_dir):
     os.makedirs(output_dir)


# start our logger, which will capture all the print statements
script_name = os.path.basename(__file__).split('.py')[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging(script_name, output_dir)



print(f"Output directory: {output_dir}")
shutil.copyfile(__file__,os.path.join(output_dir,script_name + '.py'))
logger.info(f'Original .py script file copied to output directory.')

"""_summary_

After testing at lower resolution with a subset of slices, it appears that 
1. the _v2 of cascading registration works well, but should not be done more than appx 3 times b/c of blurring
  - v2 uses do_reg_ants, which should be a drop in replacement for do_reg but is in this case MUCH better
2. interpolating missing slices is improved, but still seems to be not perfect. This can have a large impact on
the rest of the registrations
  - this still needs to be worked on. It seems to work for the cascade but now fails for the other registrations
  - fixed
"""

# 0. Convert to nifti
print('0. Converting images to .nii.gz')
logger.warning('0. Converting images to .nii.gz') #use warnings so that we can see progress on command line as well as in the log file
for idx,img_orig in enumerate(all_image_fnames):
    img = os.path.basename(img_orig).split('.')[0] 
    output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'
    
    if (os.path.isfile(output)):
        print('\t - already done, using existing image')
        if idx ==0:
            logging.warning('Looks like the first image has already been converted. All will be checked and created if missing, but there will be no further notifications here.')
        nifti = output
    else:
        print('\t - image '+str(img_orig))
        # get the TIFF image
        slice_name = str(img_orig)
        if os.path.isfile(slice_name):
            slice_img = Image.open(slice_name)
            
            slice_img = numpy.array(slice_img)
            
            ## deprecated 
            # crop: use various options, padding to ensure multiple of rescale
            # image = slice_img
            # slice_li = numpy.pad(slice_img,pad_width=((0,rescale),(0,rescale)),mode='edge')
            
            if downsample_parallel:
                slice_img = downsample_image_parallel(slice_img, rescale, n_jobs=-1)
            else:
                slice_img = downsample_image(slice_img, rescale)

            ## original approach, below, was v. slow
            ## alternative using 2d convolution to preserve cell counts (meaning is still the same here)
            # kernel = numpy.ones((rescale,rescale)) #2d convolution kernel, all 1s
            # slice_img = convolve2d(image,kernel,mode='full')[::rescale,::rescale] #can divide by rescale if we want the mean, otherwise sum is good (total cell count)

            # exceptions that need fixing, since rigid reg does not seem to address big flips
            if ('TP1' in img_orig) or ('/testProject/' in img_orig): #we have files named the same within the subdirs, so we must specify specifically the subdir (different on local vs server) 
                if 'Image_11_-_20x_01_cellCount' in img_orig:
                    slice_img = numpy.flip(slice_img,axis=0) #flip x
            
            header = nibabel.Nifti1Header()
            header.set_data_shape(slice_img.shape)
            
            # Set proper affine with resolution from the start
            affine = create_affine(slice_img.shape, voxel_res=voxel_res)
             
            nifti = nibabel.Nifti1Image(slice_img,affine=affine,header=header)
            nifti.update_header()
            save_volume(output,nifti)

        else:
            print('\tfile '+slice_name+' not found')
            
# 1. Find largeest image as baseline
print('1. Identifying the largest image to set image size')
logger.warning('1. Identifying the largest image to set image size')

largest = -1
size= 0
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]
    nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'
    shape = nibabel.load(nifti).header.get_data_shape()
    
    if shape[0]*shape[1]>size:
        size = shape[0]*shape[1]
        largest = idx
        
template = output_dir+subject+'_'+str(largest).zfill(zfill_num)+'_'+os.path.basename(all_image_fnames[largest]).split('.')[0]+'.nii.gz'    

print(f"\tUsing the following image as the template for size: {template}")

#adapt the scaling factor base on the largest image
#we work in voxel space, assuming 1x1 sizes for the slices
#we back-compute this from nighres approach, use a factor of 5 (vox_2_factor_multiplier) to relate resolution to shrinks (nd at least 5 datapoints per dimension)
vox_2_factor_multiplier = 5
initial_scaling_factor = 128
shape = nibabel.load(template).header.get_data_shape()
shape_min = min(shape)
n_scales = math.ceil(math.log(initial_scaling_factor)/math.log(2.0)) #initially set this v. large, then we choose the ones that will fit
smooth=[]
shrink=[]
for n in range(n_scales):
    smooth.append(initial_scaling_factor/math.pow(2.0,n+1))
    shrink.append(math.ceil(initial_scaling_factor/math.pow(2.0,n+1)))
num_valid_steps = numpy.where(numpy.array(shrink)*vox_2_factor_multiplier<=shape_min)[0].shape[0]
scaling_factor = int((2**num_valid_steps)/2)
# scaling_factor = num_valid_steps #scaling factor was doubling?

logger.warning(f'\tScaling factor set to: {scaling_factor}')


print('2. Bring all image slices into same place as our 2d template with an initial translation registration')
logger.warning('2. Bring all image slices into same place as our 2d template with an initial translation registration')
# initial step to bring all images into the same space of our 2d template

#check for expected output name (HARD-CODED here, will need to be changed later here and below if changed)
expected_stack_fname = f'{subject}_coreg0nl_stack.nii.gz'
if os.path.isfile(os.path.join(output_dir,expected_stack_fname)):
    logging.warning('Stack exists, skipping the current alignment step')
else:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, img in enumerate(all_image_fnames):
            img = os.path.basename(img).split('.')[0]
            nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'

            sources = [nifti]
            targets = [template]
                
            output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'_coreg0nl.nii.gz'
            
            futures.append(
                executor.submit(
                    do_initial_translation_reg, sources, targets, root_dir=output_dir, file_name=output, slice_idx_str=str(idx).zfill(zfill_num), scaling_factor=scaling_factor, mask_zero=mask_zero, voxel_res=voxel_res, use_resolution_in_registration=use_resolution_in_registration
                )    
            )

#generate a list of the current images that are now in the same space
## TODO: BEFORE HERE DOES NOT HAVE ALL THE FIXES FOR SDF
image_list = []
for idx,img_name in enumerate(all_image_fnames):
    img_name = os.path.basename(img_name).split('.')[0]
    #TODO: changed for SDF
    # image_list.append(output_dir+f'/init_translation_slice_{str(idx).zfill(zfill_num)}/'+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+'_coreg0nl_ants-def0.nii.gz')
    image_list.append(output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+'_coreg0nl_sdf_ants-def0.nii.gz')

#compute the scaling factors for sharpening from the entire dataset
sigma_multiplier, strength_multiplier, stats = compute_scaling_multipliers_from_dataset(image_list)

# deformation files are expected to be in the root output dir directly, so lets copy them here
#TODO: removed for SDF
# for _file in image_list:
#     # _def_file = os.path.basename(_file)
#     shutil.copy2(_file,output_dir)


expected_stack_name = f'{subject}_coreg0nl_stack.nii.gz'
if os.path.isfile(os.path.join(output_dir,expected_stack_name)):
    logging.warning(f'Initial stack exists, skipping the first generate_stack_and_template \n{expected_stack_name}')
else:
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag='coreg0nl_sdf',
                                           missing_idxs_to_fill=missing_idxs_to_fill,
                                           scaling_factor=scaling_factor,voxel_res=voxel_res,mask_zero=mask_zero,
                                           sigma_multiplier=sigma_multiplier,strength_multiplier=strength_multiplier,
                                           across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                           nonlin_interp_max_workers=nonlin_interp_max_workers)

## we run an initial cascading registration and allow a fair amount of warping to bring things into initial alignment
# the resulting template (which is iteratively warped with every iteration) is used to anchor the next 
# set of iterations in the STAGE1 registrations, which are more conservative

template_not_generated = True #keeps track of if we generated a template or not at this stage so that we can generate one if we stopped the registration at some point
iter_tag = ""
num_cascade_iterations = 1
anchor_slice_idxs = numpy.linspace(0,len(all_image_fnames)-1,num_cascade_iterations+2).astype(int)
anchor_slice_idxs = anchor_slice_idxs[1:-1] #remove the first and last, as they will denote 1st and last indices of the stack
for iter in range(num_cascade_iterations):
    if iter == 0:
        input_source_file_tag = 'coreg0nl_sdf'
        apply_smoothing_kernel = 0

    else:
        input_source_file_tag = iter_tag #updates with the previous iteration
        if iter == num_cascade_iterations-1:
            apply_smoothing_kernel = across_slice_smoothing_sigma #we only smooth on the last iteration
    iter_tag = f'cascade_{iter}'
    print(f'\t iteration tag: {iter_tag}')
    logger.warning('****************************************************************************')
    logger.warning(f'\titeration {iter_tag}')
    logger.warning('****************************************************************************')
    expected_stack_fname = f'{subject}_{iter_tag}_stack.nii.gz'
    logging.warning(f'====>Iteration: {iter_tag} {expected_stack_fname}')
    if os.path.isfile(os.path.join(output_dir,expected_stack_fname)):
        logging.warning('Stack exists, skipping the current cascade iteration')
        # Load the template that would have been generated
        if per_slice_template:
            # Build list of per-slice template filenames
            template = []
            for idx, img_name in enumerate(all_image_fnames):
                img_name = os.path.basename(img_name).split('.')[0]
                slice_template = output_dir + subject + '_' + str(idx).zfill(zfill_num) + '_' + img_name + f'_{iter_tag}_template.nii.gz'
                template.append(slice_template)
            if use_nonlin_slice_templates:
                template_nonlin = []
                for idx, img_name in enumerate(all_image_fnames):
                    img_name = os.path.basename(img_name).split('.')[0]
                    slice_template = output_dir + subject + '_' + str(idx).zfill(zfill_num) + '_' + img_name + f'_{iter_tag}_template_nonlin.nii.gz'
                    template_nonlin.append(slice_template)
                template = template_nonlin
        else:
            template = output_dir + subject + f'_{iter_tag}_template.nii.gz'
        template_tag = iter_tag
        template_not_generated = False
        logging.warning(f'\t\tLoaded existing template(s)')
    
    else:
        run_cascading_coregistrations_v2(output_dir, subject, 
                                    all_image_fnames, anchor_slice_idx = anchor_slice_idxs[iter], 
                                    missing_idxs_to_fill = missing_idxs_to_fill, 
                                    zfill_num=zfill_num, input_source_file_tag=input_source_file_tag, 
                                    reg_level_tag=iter_tag, previous_target_tag=None, run_syn=True,
                                    scaling_factor=scaling_factor, voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration) #,mask_zero=mask_zero)

        #we generate the template even if we do not run the registration, since we need to have a template for the next iteration
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag=iter_tag,
                                            per_slice_template=True,missing_idxs_to_fill=missing_idxs_to_fill,
                                            scaling_factor=scaling_factor,voxel_res=voxel_res,mask_zero=mask_zero,
                                            across_slice_smoothing_sigma=apply_smoothing_kernel,nonlin_interp_max_workers=nonlin_interp_max_workers)
        template_not_generated = False
        template_tag = f'cascade_{iter}' #this is to keep track of the template for subsequent reg in STEP 1

if template_not_generated:
        logging.warning('\t\tGenerating new template')
        #we generate the template even if we do not run the registration, since we need to have a template for the next iteration
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag=iter_tag,
                                            per_slice_template=True,missing_idxs_to_fill=missing_idxs_to_fill,
                                            scaling_factor=scaling_factor,voxel_res=voxel_res,mask_zero=mask_zero,
                                            across_slice_smoothing_sigma=apply_smoothing_kernel,nonlin_interp_max_workers=nonlin_interp_max_workers)
        template_tag = f'cascade_{num_cascade_iterations-1}'        

logger.warning('3. Begin STAGE1 registration iterations - Rigid + Syn')
# STEP 1: Rigid + Syn
num_reg_iterations = 5
run_rigid = True
run_syn = True
regularization ='Medium'
MI_df_struct = {} #output for MI values, will be saved in a csv file
# TODO: 2. Add masks to the registration process to improve speed (hopefully) and precision
# did this below in the groupwise reg, which decreased high frequency drift at the expense of low

## TODO: nonlin slice templates not working from cascade as of yet?
template_not_generated = True #keeps track of if we generated a template or not at this stage so that we can generate one if we stopped the registration at some point
for iter in range(num_reg_iterations):
    if iter == num_reg_iterations-1:
        across_slice_smoothing_sigma = 0 # we do not smooth the final output stack and templates
        retain_reg_mappings=True #we retain the registration mappings for all outputs at the last level
    else:
        retain_reg_mappings=False #and ONLY the last level
    #here we always go back to the original coreg0 images, we are basically just refning our target template(s) and trying not to induce too much deformation
    
    iter_tag = f"_rigsyn_{iter}"
    print(f'\t iteration tag: {iter_tag}')
    logger.warning('****************************************************************************')
    logger.warning(f'\titeration {iter_tag}')
    logger.warning('****************************************************************************')
    
    # this is currently not doing anything except computing the per-slice templates
    # logic kept for the moment.
    if (iter == 0): #do not want to use per slice templates
        # first_run_slice_template = False #skip using the per slice template on the first 2 reg steps below (up until the next template is created), same for use_nonlin_slice_templates
        first_run_slice_template = True
        first_run_nonlin_slice_template = use_nonlin_slice_templates
    else:
        first_run_slice_template = per_slice_template
        first_run_nonlin_slice_template = use_nonlin_slice_templates

    expected_stack_fname = f'{subject}_coreg12nl_win12{iter_tag}_stack.nii.gz'
    logging.warning(f'====>Iteration: {iter_tag} {expected_stack_fname}')
    if os.path.isfile(os.path.join(output_dir,expected_stack_fname)):
        logging.warning('Stack exists, skipping the current STAGE1 iteration')
        # Load the template that would have been generated
        if per_slice_template:
            # Build list of per-slice template filenames
            template = []
            for idx, img_name in enumerate(all_image_fnames):
                img_name = os.path.basename(img_name).split('.')[0]
                slice_template = output_dir + subject + '_' + str(idx).zfill(zfill_num) + '_' + img_name + f'_coreg12nl_win12{iter_tag}_template.nii.gz'
                template.append(slice_template)
            if use_nonlin_slice_templates:
                template_nonlin = []
                for idx, img_name in enumerate(all_image_fnames):
                    img_name = os.path.basename(img_name).split('.')[0]
                    slice_template = output_dir + subject + '_' + str(idx).zfill(zfill_num) + '_' + img_name + f'_coreg12nl_win12{iter_tag}_template_nonlin.nii.gz'
                    template_nonlin.append(slice_template)
                template = template_nonlin
        else:
            template = output_dir + subject + f'_coreg12nl_win12{iter_tag}_template.nii.gz'
            if use_nonlin_slice_templates:
                template_nonlin = output_dir + subject + f'_coreg12nl_win12{iter_tag}_template_nonlin.nii.gz'
                template = template_nonlin
        template_tag = 'coreg12nl_win12' + iter_tag
        template_not_generated = False
        logging.warning(f'\t\tLoaded existing template(s)')
    else:

        slice_offset_list_forward = [-3,-2,-1,] #weighted back
        slice_offset_list_reverse = [1,2,3] #weighted forward
        image_weights_win1 = generate_gaussian_weights([0,] + slice_offset_list_forward, gauss_std=3) #symmetric gaussian, so the same on both sides
        image_weights_win2 = generate_gaussian_weights([0,] + slice_offset_list_reverse, gauss_std=3)

        run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers, 
                                    target_slice_offset_list=slice_offset_list_forward, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl_sdf', 
                                    reg_level_tag='coreg1nl'+iter_tag,
                                    image_weights=image_weights_win1,
                                    run_syn=run_syn,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration)
        run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers, 
                                    target_slice_offset_list=slice_offset_list_reverse, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl_sdf', 
                                    reg_level_tag='coreg2nl'+iter_tag,
                                    image_weights=image_weights_win2,
                                    run_syn=run_syn,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration)

        logging.warning('\t\tSelecting best registration by MI')

        select_best_reg_by_MI_parallel(output_dir,subject,all_image_fnames,template_tag=template_tag,
                            zfill_num=zfill_num,reg_level_tag1='coreg1nl'+iter_tag, reg_level_tag2='coreg2nl'+iter_tag,
                            reg_output_tag='coreg12nl'+iter_tag,per_slice_template=first_run_slice_template,df_struct=MI_df_struct,
                            use_nonlin_slice_templates=first_run_nonlin_slice_template,max_workers=max_workers)
        if MI_df_struct is not None:
            pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
        
        logging.warning('\t\tGenerating new template')
        if 'nonlin' in slice_template_type:
            template, template_nonlin = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                                zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                                missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                                scaling_factor=scaling_factor, nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                mask_zero=mask_zero,across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                                voxel_res=voxel_res)
        else:
            template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                                zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                                missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                                scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                mask_zero=mask_zero,across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                                voxel_res=voxel_res)
        if use_nonlin_slice_templates:
            template = template_nonlin
        # missing_idxs_to_fill = None #if we only want to fill in missing slices on the first iteration, then we just use that image as the template
        
        ## TODO: insert in here the code to register the stack to the MRI template and then update the tag references as necessary
        # if iter > 0: #we do not do this on the first iteration
            # MRI_reg_output = register_stack_to_mri(slice_stack_template, mri_template)


        template_tag = 'coreg12nl'+iter_tag

        # her we include neigbouring slices and increase the sharpness of the gaussian
        slice_offset_list_forward = [-6,-5,-4,-3,-2,-1,1,2,3] #weighted back, but also forward
        slice_offset_list_reverse = [-3,-2,-1,1,2,3,4,5,6] #weighted forward, but also back

        
        image_weights_win1 = generate_gaussian_weights([0,] + slice_offset_list_forward, gauss_std=4) #symmetric gaussian, so the same on both sides
        image_weights_win2 = generate_gaussian_weights([0,] + slice_offset_list_reverse, gauss_std=4)

        run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers,
                                    target_slice_offset_list=slice_offset_list_forward, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl_sdf', 
                                    previous_target_tag = None,
                                    reg_level_tag='coreg12nl_win1'+iter_tag,
                                    image_weights=image_weights_win1,
                                    run_syn=run_syn,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    mask_zero=mask_zero,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration)
        
        run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers,
                                    target_slice_offset_list=slice_offset_list_reverse, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl_sdf',
                                    previous_target_tag = None,
                                    reg_level_tag='coreg12nl_win2'+iter_tag,
                                    image_weights=image_weights_win2,
                                    run_syn=run_syn,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    mask_zero=mask_zero,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration)
        logging.warning('\t\tSelecting best registration by MI')                                     

        
        logging.warning(template_tag)
        # logging.warning(f'coreg12nl_win1{iter_tag}')
        ## ERROR HERE in the next line ###
        select_best_reg_by_MI_parallel(output_dir,subject,all_image_fnames,template_tag=template_tag,
                            zfill_num=zfill_num,reg_level_tag1='coreg12nl_win1'+iter_tag, reg_level_tag2='coreg12nl_win2'+iter_tag,
                            reg_output_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,df_struct=MI_df_struct,
                            use_nonlin_slice_templates=use_nonlin_slice_templates,max_workers=max_workers)
        if MI_df_struct is not None:
            pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
        
        logging.warning('\t\tGenerating new template')
        if 'nonlin' in slice_template_type:
            template, template_nonlin = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                                zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                                missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                                scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                                voxel_res=voxel_res)
        else:
            template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                                zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                                missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                                scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                                voxel_res=voxel_res)
        
        template_not_generated = False
        if use_nonlin_slice_templates:
            template = template_nonlin
        template_tag = 'coreg12nl_win12'+iter_tag


if template_not_generated:
    logging.warning('\t\tGenerating new template')
    if 'nonlin' in slice_template_type:
        template, template_nonlin = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                            zfill_num=zfill_num,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                            missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                            scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                            across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                            voxel_res=voxel_res)
    else:
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                            zfill_num=zfill_num,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                            missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                            scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                            across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                            voxel_res=voxel_res)
    template_tag = 'coreg12nl_win12'+f'_rigsyn_{num_reg_iterations-1}'

final_rigsyn_reg_level_tag = template_tag


# Use the final template from Rigid + Syn as the input for Syn only
input_source_file_tag = final_rigsyn_reg_level_tag

# After final Rigid + Syn iterations
logging.warning("=" * 80)
logging.warning("STARTING GROUPWISE OPTIMIZATION - This will reduce wave artifacts")
logging.warning("=" * 80)

# optmized_stack = groupwise_stack_optimization(
#     output_dir, subject, all_image_fnames,
#     reg_level_tag=f'{input_source_file_tag}',
#     iterations=5,
#     scaling_factor=scaling_factor,
#     voxel_res=voxel_res
# )

groupwise_iterations = 10
logging.warning(f'Scaling factor: {scaling_factor}')
groupwise_stack_optimization_embedded_antspy(output_dir,subject, all_image_fnames, 
                                reg_level_tag=f'{input_source_file_tag}',
                                iterations=groupwise_iterations,
                                zfill_num=4,
                                scaling_factor=scaling_factor, 
                                use_resolution_in_registration=use_resolution_in_registration,
                                max_workers=max_workers,
                                use_deformed_source_after_iteration=4,
                                use_local_template_after_iteration=3,
                                local_template_window=5)


#we generate all the stacks at the end, since we will need to identify which is potentially the best 
# - smooth transitions without much/any shape change
# we output the template with 'nochange' to simply stack the def0 files without modification

logging.warning('\t\tGenerating new stacks after groupwise optimization')
for iter in range(groupwise_iterations):
    template = generate_stack_and_template(
        output_dir, subject, all_image_fnames,
        zfill_num=zfill_num,
        reg_level_tag=f'{input_source_file_tag}'+ f'_groupwise_iter{iter}',
        per_slice_template=False, #we do not need output templates
        missing_idxs_to_fill=missing_idxs_to_fill,
        slice_template_type='nochange',
        scaling_factor=scaling_factor,
        nonlin_interp_max_workers=nonlin_interp_max_workers,
        voxel_res=voxel_res
    )

logging.warning(f"Output directory: {output_dir}")