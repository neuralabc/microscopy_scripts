"""Application script for running 2D slice registration on specific dataset.

Local copy of run_slice_registration_optimized_newReg_sdf_v2.py for this workstation (uv env, data on /mnt/ssd).
Run from a scratch working directory (ANTs can leave files in the cwd), e.g.:
    cd <scratch> && uv run --project /home/chris/Documents/code/microscopy_scripts \
        python /home/chris/Documents/code/microscopy_scripts/macaque_CB/run_slice_registration_optimized_newReg_sdf_v2_local.py
See REGISTRATION_NOTES_local.md for the SLICEREG_* options. Set SLICEREG_SMOKE=1 for a quick test (first 7 slices, slice 3 treated as missing, rescale 40).
"""
import env_compat  # must come before slice_registration_functions (threads, OOM priority, ANTsPy/nighres shims)
from slice_registration_functions import *
from slice_structure_identification_functions import compute_signed_distance_weight_filled
from slice_registration_extensions import run_parallel_coregistrations_v3, groupwise_stack_optimization_embedded_antspy_v2
import json
import subprocess
import time

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
use_signed_distance_weighting_for_registration = False #compute signed distance function from the images and use this image for registration

# scaling factor that is applied to the x and y dimensions (in-plane dimensions) to downsample the data
# rescale=5 #larger scale means that you have to change the scaling_factor, which is now done automatically just before computations
rescale=int(os.environ.get('SLICEREG_RESCALE', 5))
#rescale=40
smoke_test = os.environ.get('SLICEREG_SMOKE', '0') == '1'
if smoke_test:
    rescale = 40

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
# local workstation (24 cores, shared with other jobs): keep workers low, each worker is single-threaded (see env_compat)
max_workers = int(os.environ.get('SLICEREG_MAX_WORKERS', 8))
nonlin_interp_max_workers = max_workers #number of workers to use for nonlinear slice interpolation when use_nonlin_slice_templates = True

# STAGE1 neighbour targets (see slice_registration_extensions.py):
#  'coreg0nl' (original behaviour): neighbours are the translation-only slices in every iteration
#  'previous': neighbours are the deformed slices of the level that also provides the template (cascade_0, then the previous STAGE1 output)
# the source is coreg0nl in both cases, so the outputs and the transform chain are unchanged
stage1_neighbour_targets = os.environ.get('SLICEREG_NEIGHBOUR_TARGETS', 'coreg0nl')
if stage1_neighbour_targets not in ['coreg0nl', 'previous']:
    raise ValueError(f'Unknown stage1_neighbour_targets: {stage1_neighbour_targets}')
variant_tag = '' if stage1_neighbour_targets == 'coreg0nl' else '_nbrprev'

# rigid jitter correction folded into coreg0nl (see slice_jitter.py): each slice is fitted rigidly to the mean of its
# neighbours, only the fast part of the poses is kept (the slow path stays as it was), iterated with half steps.
# The coreg0nl maps, inverse maps and deformed slices are rewritten in place (same names and formats), so the
# transform chain is unchanged; the poses are in {subject}_coreg0nl_jitter_poses.csv
# default on since 2026-10-28 (better than without at rescale 40 and 20, REGISTRATION_NOTES_local.md 4.7); the output
# folder keeps the '_jit' suffix, so it never mixes with the older folders made without the correction
use_jitter_correction = os.environ.get('SLICEREG_JITTER', '1') == '1'
jitter_sigma = float(os.environ.get('SLICEREG_JITTER_SIGMA', 15)) #slices; the slow part removed from the poses
jitter_iterations = int(os.environ.get('SLICEREG_JITTER_ITERS', 6))
if use_jitter_correction:
    variant_tag += '_jit' + ('' if jitter_iterations == 6 else str(jitter_iterations))
# number of groupwise iterations; the final level (and the end of the transform chain) is groupwise_iter{N-1}
groupwise_iterations = int(os.environ.get('SLICEREG_GROUPWISE_ITERS', 10))
if groupwise_iterations != 10:
    variant_tag += f'_gw{groupwise_iterations}'
if os.environ.get('SLICEREG_STAGE1_SYN', '1') != '1':
    variant_tag += '_s1rigid'
elif os.environ.get('SLICEREG_STAGE1_SYN_LAST', '1') != '1':
    variant_tag += '_s1lastrigid'
if os.environ.get('SLICEREG_STAGE1_REG', 'Medium') != 'Medium':
    variant_tag += '_s1reg' + os.environ['SLICEREG_STAGE1_REG']

def stage1_neighbour_target_tag(template_tag):
    return template_tag if stage1_neighbour_targets == 'previous' else None

# STAGE1 MRI anchor (see mri_anchor.py): replace the whole-stack median target of every slice with the plane of
# the MRI template that corresponds to it. The MRI is fitted to the current stack with a shear-free affine
# at each template level, so it corrects low-frequency drift and global shape but not folium-level detail.
# The source is still coreg0nl, so the outputs and the transform chain are unchanged; new files are
# *_{level}_mri_target.nii.gz and mri_anchor/{level}/ (transforms + qc.json)
use_mri_anchor = os.environ.get('SLICEREG_MRI_ANCHOR', '0') == '1'
mri_target_weight_scale = float(os.environ.get('SLICEREG_MRI_WEIGHT', 1.0)) #multiplies the weight of the (replaced) stack template
mri_refit = os.environ.get('SLICEREG_MRI_REFIT', '0') == '1' #1: refit the MRI affine at every STAGE1 level (warm start); 0: fit once on cascade_0 and keep it
# refitting let the reference follow the stack (run C, 2026-10-27: Dice 0.86 -> 0.76, section scale 1.01 -> 1.20 over the levels)
mri_fixed_level = 'cascade_0'
mri_direction = int(os.environ.get('SLICEREG_MRI_DIRECTION', 1)) #+1/-1: hemisphere of the MRI that the stack is fitted to (the MRI is near symmetric; fixed so that runs are comparable)
mri_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_extras', 'cerebellum_target')
mri_template_fname = os.path.join(mri_dir, 'D99_cerebellum_brainstem_template_crop.nii.gz')
mri_mask_fname = os.path.join(mri_dir, 'D99_cerebellum_brainstem_mask_crop.nii.gz')
if not use_mri_anchor:
    mri_target_weight_scale = 1.0 #only applies to the MRI targets
if use_mri_anchor:
    import json
    import mri_anchor
    variant_tag += ('_mri' if mri_refit else '_mrifix') + ('' if mri_target_weight_scale == 1.0 else f'{mri_target_weight_scale:g}')
mri_state = None #qc of the last MRI fit, used to warm-start the next one

def stage1_stack_targets(template_tag):
    """Per-slice MRI targets for the stack at `template_tag` (computed once per level), or None."""
    global mri_state
    if not use_mri_anchor:
        return None
    if not mri_refit:
        template_tag = mri_fixed_level #one fit, the same MRI planes for every level
    fnames = [f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{os.path.basename(f).split('.')[0]}_{template_tag}_mri_target.nii.gz"
              for idx, f in enumerate(all_image_fnames)]
    level_dir = os.path.join(output_dir, 'mri_anchor', template_tag)
    qc_fname = os.path.join(level_dir, 'qc.json')
    if all(os.path.isfile(f) for f in fnames) and os.path.isfile(qc_fname):
        with open(qc_fname) as f:
            mri_state = json.load(f)
        return fnames
    stack = nibabel.load(f'{output_dir}{subject}_{template_tag}_stack.nii.gz').get_fdata().astype(numpy.float32)
    res_mm = rescaled_in_plane_res_x / 1000.0
    slice_mm = in_plane_res_z_microns / 1000.0
    aff_f, qc = mri_anchor.stack_to_mri_affine(stack, res_mm, slice_mm, mri_template_fname, mri_mask_fname,
                                               level_dir, init_state=mri_state, direction=mri_direction)
    mri_anchor.write_mri_slice_targets(aff_f, stack.shape, res_mm, slice_mm, mri_template_fname, fnames, voxel_res)
    with open(qc_fname, 'w') as f:
        json.dump(qc, f, indent=2, default=str)
    logging.warning(f'MRI targets for {template_tag}: dice_affine={qc["dice_affine"]:.3f}, scales={qc["scales_stack_axes"]}')
    mri_state = qc
    return fnames

# setup the output directory for use
if use_signed_distance_weighting_for_registration:
    sdf_tag = '_sdf'
    cost_function = 'CrossCorrelation'
    missing_slice_interp_method = 'mean'
    cortical_detail_weight = .3 #.5 equally balances cortical ribbon with inside definition of structure (.5 is equal weighting, .3 is good for stability I think!), smaller values preserve more of the interior holes but may lose some cortical detail, larger values preserve more of the cortical detail but may lose definition in the interior (since it will blend with the filled SDF)
    if rescale == 40:
        sdf_clip_val = 10
    elif rescale == 5:
        sdf_clip_val = 100
    else:
        sdf_clip_val = 50
else:
    sdf_tag = ''
    cost_function = 'MutualInformation'
    missing_slice_interp_method = 'intermediate_nonlin_mean'
    sdf_clip_val = 0
    cortical_detail_weight = None #not used when not SDF

# local data: the lookup csv still lists the old /data/data_drive mount, which is now /mnt/ssd
data_root = '/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/'
output_dir = f'{data_root}{subject}_sliceReg_optimized_v2_rescale_{rescale}{sdf_tag}_local{variant_tag}{"_smoke" if smoke_test else ""}/'
_df = pd.read_csv(f'{data_root}all_TP_image_idxs_file_lookup.csv')
_df['file_name'] = _df['file_name'].str.replace('/data/data_drive/', '/mnt/ssd/', regex=False)

missing_idxs_to_fill = [32,59,120,160,189,228] #these are the slice indices with missing or terrible data, fill with coreg of neighbours
# output_dir = '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/slice_reg_perSliceTemplate_image_weights_all_tmp/'
## _df = pd.read_csv('/data/data_drive/Macaque_CB/processing/results_from_cell_counts/all_TP_image_idxs_file_lookup.csv')

#missing_idxs_to_fill = [32]
# missing_idxs_to_fill = [5]
# missing_idxs_to_fill = None
all_image_fnames = list(_df['file_name'].values)
if smoke_test:
    all_image_fnames = all_image_fnames[0:7]
    missing_idxs_to_fill = [3] #exercise the missing-slice interpolation path

# ## for testing XXX
# all_image_fnames = all_image_fnames[0:5] #for testing
# all_image_fnames = all_image_fnames[0:35] #for testing
# missing_idxs_to_fill = None
# missing_idxs_to_fill = [missing_idxs_to_fill[0]]

print('*********************************************************************************************************')
print(f'Output directory: {output_dir}')
print('*********************************************************************************************************')

# set missing indices, which will be iteratively filled with the mean of the neighbouring slices
if missing_idxs_to_fill is not None:
    if numpy.max(numpy.array(missing_idxs_to_fill)) > len(all_image_fnames): #since these are indices, will start @ 0
        raise ValueError("Missing slice indices exceed the number of images in the stack.")

# all_image_fnames = all_image_fnames[0:10] #for testing
# all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below


if not os.path.exists(output_dir):
     os.makedirs(output_dir)


# start our logger, which will capture all the print statements
script_name = os.path.basename(__file__).split('.py')[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging(script_name, output_dir)

# settings of this run, for the QC report and for later reference (a new file; no existing output changes)
def _git_commit():
    try:
        import subprocess
        return subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'
with open(os.path.join(output_dir, 'run_settings.json'), 'w') as _fh:
    json.dump(dict(script=os.path.abspath(__file__), git_commit=_git_commit(), started=time.strftime('%Y-%m-%d %H:%M'),
                   subject=subject, rescale=rescale, in_plane_res_um=[rescaled_in_plane_res_x, rescaled_in_plane_res_y],
                   max_workers=max_workers, missing_idxs_to_fill=missing_idxs_to_fill, variant_tag=variant_tag,
                   jitter_correction=use_jitter_correction, jitter_sigma=jitter_sigma, jitter_iterations=jitter_iterations,
                   groupwise_iterations=groupwise_iterations,
                   env={k: v for k, v in sorted(os.environ.items()) if k.startswith('SLICEREG_')}), _fh, indent=1)



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
                slice_img = downsample_image(slice_img, rescale, pad_value=-1*sdf_clip_val) #this sets the pad value to be a negative value outside of the range of the data, which is important for the SDF computation later, since we want to make sure that the padding does not create artificial boundaries that will impact the SDF computation (since it will create a boundary between the data and the padding)

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

            if use_signed_distance_weighting_for_registration:
                # we want a "raw" SDF , then we need to turn off the sigma weights and clip the values to focus on the boundary
                # we also need to change the similarity metric, since we have directionality
                logging.warning(f'Computing SDF for registration: {output}')
                if rescale == 5:
                    sdf = compute_signed_distance_weight_filled(nifti, img_smth_gauss=1,pctl_cut=95, closing_radius=2, 
                                                         smooth_weights_sigma=None, min_object_size=100, clip=sdf_clip_val,
                                                         cortical_detail_weight=cortical_detail_weight)
                elif rescale == 40:
                    sdf = compute_signed_distance_weight_filled(nifti, img_smth_gauss=0,pctl_cut=95, closing_radius=0, 
                                                         smooth_weights_sigma=None, min_object_size=5, clip=sdf_clip_val,
                                                         cortical_detail_weight=cortical_detail_weight)
                save_volume(output.replace('.nii.gz','_orig.nii.gz'),nifti)    #save the original as _orig
                nifti = sdf #save the sdf
            save_volume(output,nifti)

        else:
            print('\tfile '+slice_name+' not found')

# 1. Find largest image as baseline
print('1. Identifying the largest image to set image size')
logger.warning('1. Identifying the largest image to set image size')

largest = -1
size= 0
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]
    nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'
    # Use nibabel directly for faster header-only reading (no full data load)
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
                    do_initial_translation_reg, sources, targets, root_dir=output_dir, file_name=output, slice_idx_str=str(idx).zfill(zfill_num), scaling_factor=scaling_factor, mask_zero=mask_zero, voxel_res=voxel_res, use_resolution_in_registration=use_resolution_in_registration, cost_function=cost_function, default_value=-1*sdf_clip_val
                )    
            )

#generate a list of the current images that are now in the same space
image_list = []
for idx,img_name in enumerate(all_image_fnames):
    img_name = os.path.basename(img_name).split('.')[0]
    image_list.append(output_dir+f'/init_translation_slice_{str(idx).zfill(zfill_num)}/'+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+'_coreg0nl_ants-def0.nii.gz')
    
#compute the scaling factors for sharpening from the entire dataset
sigma_multiplier, strength_multiplier, stats = compute_scaling_multipliers_from_dataset(image_list)

# deformation files are expected to be in the root output dir directly, so lets copy them here (skip if already exists)
for _file in image_list:
    dest_file = os.path.join(output_dir, os.path.basename(_file))
    if not os.path.isfile(dest_file):
        shutil.copy2(_file, output_dir)

if use_jitter_correction:
    jitter_poses_fname = os.path.join(output_dir, f'{subject}_coreg0nl_jitter_poses.csv')
    if os.path.isfile(jitter_poses_fname): #written last, so its presence means the correction is complete
        logging.warning('Jitter correction of coreg0nl already done, skipping')
    else:
        logger.warning('2b. Rigid jitter correction of coreg0nl (slow path kept)')
        import slice_jitter
        import mri_anchor
        slice_jitter.correct_level_in_place(output_dir, subject, all_image_fnames, 'coreg0nl',
                                            res_mm=rescaled_in_plane_res_x / 1000.0, voxel_res=voxel_res,
                                            thr=mri_anchor.tissue_threshold_for(rescaled_in_plane_res_x / 1000.0),
                                            sigma=jitter_sigma, iters=jitter_iterations, step=0.5,
                                            exclude=missing_idxs_to_fill or [], max_workers=max_workers,
                                            zfill_num=zfill_num)


expected_stack_name = f'{subject}_coreg0nl_stack.nii.gz'
if os.path.isfile(os.path.join(output_dir,expected_stack_name)):
    logging.warning(f'Initial stack exists, skipping the first generate_stack_and_template \n{expected_stack_name}')
else:
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag='coreg0nl',
                                           missing_idxs_to_fill=missing_idxs_to_fill,
                                           scaling_factor=scaling_factor,voxel_res=voxel_res,mask_zero=mask_zero,
                                           sigma_multiplier=sigma_multiplier,strength_multiplier=strength_multiplier,
                                           across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                           nonlin_interp_max_workers=nonlin_interp_max_workers,
                                           missing_slice_interp_method=missing_slice_interp_method)

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
        input_source_file_tag = 'coreg0nl'
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
                                    use_resolution_in_registration=use_resolution_in_registration,
                                    default_value=-1*sdf_clip_val) #,mask_zero=mask_zero)

        #we generate the template even if we do not run the registration, since we need to have a template for the next iteration
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag=iter_tag,
                                            per_slice_template=True,missing_idxs_to_fill=missing_idxs_to_fill,
                                            scaling_factor=scaling_factor,voxel_res=voxel_res,mask_zero=mask_zero,
                                            across_slice_smoothing_sigma=apply_smoothing_kernel,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                            missing_slice_interp_method=missing_slice_interp_method)
        template_not_generated = False
        template_tag = f'cascade_{iter}' #this is to keep track of the template for subsequent reg in STEP 1

if template_not_generated:
        logging.warning('\t\tGenerating new template')
        #we generate the template even if we do not run the registration, since we need to have a template for the next iteration
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag=iter_tag,
                                            per_slice_template=True,missing_idxs_to_fill=missing_idxs_to_fill,
                                            scaling_factor=scaling_factor,voxel_res=voxel_res,mask_zero=mask_zero,
                                            across_slice_smoothing_sigma=apply_smoothing_kernel,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                            missing_slice_interp_method=missing_slice_interp_method)
        template_tag = f'cascade_{num_cascade_iterations-1}'        

logger.warning('3. Begin STAGE1 registration iterations - Rigid + Syn')
# STEP 1: Rigid + Syn
num_reg_iterations = 5
run_rigid = True
run_syn = os.environ.get('SLICEREG_STAGE1_SYN', '1') == '1' #0: STAGE1 rigid only, all warping left to the groupwise step
stage1_syn_last = os.environ.get('SLICEREG_STAGE1_SYN_LAST', '1') == '1' #0: SyN only in the template-building STAGE1 levels, the last level (in the chain) rigid only
regularization = os.environ.get('SLICEREG_STAGE1_REG', 'Medium') #SyN preset in embedded_antspy_smthMod: Low, Medium, High, VeryHigh
MI_df_struct = {} #output for MI values, will be saved in a csv file
# TODO: 2. Add masks to the registration process to improve speed (hopefully) and precision
# did this below in the groupwise reg, which decreased high frequency drift at the expense of low

## TODO: nonlin slice templates not working from cascade as of yet?
template_not_generated = True #keeps track of if we generated a template or not at this stage so that we can generate one if we stopped the registration at some point
for iter in range(num_reg_iterations):
    # the last STAGE1 level is the one that goes into the groupwise step and the transform chain; earlier levels
    # only build templates (every level registers from coreg0nl again)
    run_syn_this_iter = run_syn and not (iter == num_reg_iterations-1 and not stage1_syn_last)
    if iter == num_reg_iterations-1:
        across_slice_smoothing_sigma = 0 # we do not smooth the final output stack and templates
        retain_reg_mappings=True #we retain the registration mappings for all outputs at the last level
    else:
        retain_reg_mappings=True #and ONLY the last level XXX SET TO TRUE FOR TESTING
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

        run_parallel_coregistrations_v3(output_dir, subject, all_image_fnames, template, max_workers=max_workers, 
                                    target_slice_offset_list=slice_offset_list_forward, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl', 
                                    reg_level_tag='coreg1nl'+iter_tag,
                                    image_weights=image_weights_win1,
                                    run_syn=run_syn_this_iter,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration,
                                    cost_function=cost_function,
                                    default_value=-1*sdf_clip_val,
                                    neighbour_target_tag=stage1_neighbour_target_tag(template_tag),
                                    stack_template_list=stage1_stack_targets(template_tag),
                                    stack_template_weight_scale=mri_target_weight_scale)
        run_parallel_coregistrations_v3(output_dir, subject, all_image_fnames, template, max_workers=max_workers, 
                                    target_slice_offset_list=slice_offset_list_reverse, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl', 
                                    reg_level_tag='coreg2nl'+iter_tag,
                                    image_weights=image_weights_win2,
                                    run_syn=run_syn_this_iter,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration,
                                    cost_function=cost_function,
                                    default_value=-1*sdf_clip_val,
                                    neighbour_target_tag=stage1_neighbour_target_tag(template_tag),
                                    stack_template_list=stage1_stack_targets(template_tag),
                                    stack_template_weight_scale=mri_target_weight_scale)

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
                                                voxel_res=voxel_res,missing_slice_interp_method=missing_slice_interp_method)
        else:
            template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                                zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                                missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                                scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                mask_zero=mask_zero,across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                                voxel_res=voxel_res,missing_slice_interp_method=missing_slice_interp_method)
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

        run_parallel_coregistrations_v3(output_dir, subject, all_image_fnames, template, max_workers=max_workers,
                                    target_slice_offset_list=slice_offset_list_forward, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl', 
                                    previous_target_tag = None,
                                    reg_level_tag='coreg12nl_win1'+iter_tag,
                                    image_weights=image_weights_win1,
                                    run_syn=run_syn_this_iter,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    mask_zero=mask_zero,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration,
                                    cost_function=cost_function,
                                    default_value=-1*sdf_clip_val,
                                    neighbour_target_tag=stage1_neighbour_target_tag(template_tag),
                                    stack_template_list=stage1_stack_targets(template_tag),
                                    stack_template_weight_scale=mri_target_weight_scale)
        
        run_parallel_coregistrations_v3(output_dir, subject, all_image_fnames, template, max_workers=max_workers,
                                    target_slice_offset_list=slice_offset_list_reverse, 
                                    zfill_num=zfill_num, 
                                    input_source_file_tag='coreg0nl',
                                    previous_target_tag = None,
                                    reg_level_tag='coreg12nl_win2'+iter_tag,
                                    image_weights=image_weights_win2,
                                    run_syn=run_syn_this_iter,
                                    run_rigid=run_rigid,
                                    scaling_factor=scaling_factor,
                                    mask_zero=mask_zero,
                                    regularization=regularization,
                                    retain_reg_mappings=retain_reg_mappings,
                                    voxel_res=voxel_res,
                                    use_resolution_in_registration=use_resolution_in_registration,
                                    cost_function=cost_function,
                                    default_value=-1*sdf_clip_val,
                                    neighbour_target_tag=stage1_neighbour_target_tag(template_tag),
                                    stack_template_list=stage1_stack_targets(template_tag),
                                    stack_template_weight_scale=mri_target_weight_scale)
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
                                                voxel_res=voxel_res,missing_slice_interp_method=missing_slice_interp_method)
        else:
            template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                                zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                                missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                                scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                                voxel_res=voxel_res,missing_slice_interp_method=missing_slice_interp_method)
        
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
                                            voxel_res=voxel_res,missing_slice_interp_method=missing_slice_interp_method)
    else:
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                            zfill_num=zfill_num,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                            missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                            scaling_factor=scaling_factor,nonlin_interp_max_workers=nonlin_interp_max_workers,
                                            across_slice_smoothing_sigma=across_slice_smoothing_sigma,
                                            voxel_res=voxel_res,missing_slice_interp_method=missing_slice_interp_method)
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

expected_stack_fname = f'{subject}_{input_source_file_tag}_groupwise_iter{groupwise_iterations-1}_stack.nii.gz'
logging.warning(expected_stack_fname)
if os.path.isfile(os.path.join(output_dir,expected_stack_fname)):
    logging.warning('Groupwise optimization stacks exist, skipping the groupwise optimization step')
else:
    groupwise_stack_optimization_embedded_antspy_v2(output_dir,subject, all_image_fnames, 
                                    reg_level_tag=f'{input_source_file_tag}',
                                    iterations=groupwise_iterations,
                                    zfill_num=4,
                                    scaling_factor=scaling_factor, 
                                    use_resolution_in_registration=use_resolution_in_registration,
                                    max_workers=max_workers,
                                    use_deformed_source_after_iteration=4,
                                    use_local_template_after_iteration=3,
                                    local_template_window=5,
                                    cost_function=cost_function)


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
        voxel_res=voxel_res,
        missing_slice_interp_method=missing_slice_interp_method
    )

logging.warning(f"Output directory: {output_dir}")

# Generate CSV files with all map.nii.gz files for each slice (separate files for forward and inverse maps)

fwd_csv_path = os.path.join(output_dir, f'{subject}_forward_maps.csv')
inv_csv_path = os.path.join(output_dir, f'{subject}_inverse_maps.csv')

if os.path.isfile(fwd_csv_path) and os.path.isfile(inv_csv_path):
    logging.warning('CSV files with deformation map file paths already exist, skipping generation')
else:
    logging.warning('\t\tGenerating CSV files with deformation map file paths')
    import re  # for parsing iteration numbers from filenames
    fwd_map_data = {}  # forward maps
    inv_map_data = {}  # inverse maps

    for idx, img_name in enumerate(all_image_fnames):
        img_basename = os.path.basename(img_name).split('.')[0]
        slice_prefix = f"{subject}_{str(idx).zfill(zfill_num)}_{img_basename}"
        
        # Find all map files for this slice (handle case where none exist)
        try:
            map_files = glob.glob(os.path.join(output_dir, f"**/*{slice_prefix}*map*.nii.gz"), recursive=True)
            map_files += glob.glob(os.path.join(output_dir, f"{slice_prefix}*map*.nii.gz"))
            map_files = list(set(map_files))  # Remove duplicates
        except Exception as e:
            logging.warning(f"Could not search for map files for slice {idx}: {e}")
            map_files = []
        
        # Initialize slice entries
        if idx not in fwd_map_data:
            fwd_map_data[idx] = {'slice_idx': idx, 'slice_name': img_basename}
        if idx not in inv_map_data:
            inv_map_data[idx] = {'slice_idx': idx, 'slice_name': img_basename}
        
        # Categorize map files by transform type and iteration
        for map_file in map_files:
            # Verify file exists before adding
            if not os.path.isfile(map_file):
                continue
                
            fname = os.path.basename(map_file)
            
            # Determine map type (forward or inverse)
            is_inverse = 'invmap' in fname
            
            # Extract iteration/stage info from filename to build column name (without map_type suffix)
            # Check for groupwise iterations first
            if 'groupwise_iter' in fname:
                match = re.search(r'groupwise_iter(\d+)', fname)
                if match:
                    col_name = f"groupwise_iter{match.group(1)}"
                else:
                    col_name = f"groupwise"
            # Check for rigsyn iterations with window variants
            elif 'coreg12nl_win12' in fname:
                match = re.search(r'_rigsyn_(\d+)', fname)
                iter_str = f"_iter{match.group(1)}" if match else ""
                col_name = f"rigsyn{iter_str}_win12"
            elif 'coreg12nl_win1' in fname:
                match = re.search(r'_rigsyn_(\d+)', fname)
                iter_str = f"_iter{match.group(1)}" if match else ""
                col_name = f"rigsyn{iter_str}_win1"
            elif 'coreg12nl_win2' in fname:
                match = re.search(r'_rigsyn_(\d+)', fname)
                iter_str = f"_iter{match.group(1)}" if match else ""
                col_name = f"rigsyn{iter_str}_win2"
            elif 'coreg12nl' in fname:
                match = re.search(r'_rigsyn_(\d+)', fname)
                iter_str = f"_iter{match.group(1)}" if match else ""
                col_name = f"rigsyn{iter_str}"
            elif 'coreg1nl' in fname:
                match = re.search(r'_rigsyn_(\d+)', fname)
                iter_str = f"_iter{match.group(1)}" if match else ""
                col_name = f"coreg1nl{iter_str}"
            elif 'coreg2nl' in fname:
                match = re.search(r'_rigsyn_(\d+)', fname)
                iter_str = f"_iter{match.group(1)}" if match else ""
                col_name = f"coreg2nl{iter_str}"
            elif 'coreg0nl' in fname:
                col_name = f"coreg0nl"
            elif 'cascade' in fname:
                match = re.search(r'cascade_(\d+)', fname)
                if match:
                    col_name = f"cascade_iter{match.group(1)}"
                else:
                    col_name = f"cascade"
            else:
                # Try to extract any iteration number from filename
                iter_match = re.search(r'iter(\d+)', fname)
                iter_str = f"_iter{iter_match.group(1)}" if iter_match else ""
                col_name = f"other{iter_str}"
            
            # Add to appropriate dictionary
            if is_inverse:
                inv_map_data[idx][col_name] = map_file
            else:
                fwd_map_data[idx][col_name] = map_file

    # Process and save forward maps
    fwd_df = pd.DataFrame.from_dict(fwd_map_data, orient='index')
    fwd_df = fwd_df.sort_values('slice_idx').reset_index(drop=True)

    # Remove empty columns for forward maps
    fwd_map_cols = [c for c in fwd_df.columns if c not in ['slice_idx', 'slice_name']]
    fwd_empty_cols = [c for c in fwd_map_cols if fwd_df[c].isna().all()]
    if fwd_empty_cols:
        logging.warning(f"Removing {len(fwd_empty_cols)} empty forward map columns")
        fwd_df = fwd_df.drop(columns=fwd_empty_cols)

    # Reorder columns
    fwd_cols_sorted = sorted([c for c in fwd_df.columns if c not in ['slice_idx', 'slice_name']])
    fwd_df = fwd_df[['slice_idx', 'slice_name'] + fwd_cols_sorted]

    # Save forward maps CSV
    fwd_df.to_csv(fwd_csv_path, index=False)
    num_fwd_maps = fwd_df[fwd_cols_sorted].notna().sum().sum() if fwd_cols_sorted else 0
    logging.warning(f"Forward map CSV saved to: {fwd_csv_path}")
    logging.warning(f"  Found {num_fwd_maps} forward maps across {len(fwd_cols_sorted)} transform types")

    # Process and save inverse maps
    inv_df = pd.DataFrame.from_dict(inv_map_data, orient='index')
    inv_df = inv_df.sort_values('slice_idx').reset_index(drop=True)

    # Remove empty columns for inverse maps
    inv_map_cols = [c for c in inv_df.columns if c not in ['slice_idx', 'slice_name']]
    inv_empty_cols = [c for c in inv_map_cols if inv_df[c].isna().all()]
    if inv_empty_cols:
        logging.warning(f"Removing {len(inv_empty_cols)} empty inverse map columns")
        inv_df = inv_df.drop(columns=inv_empty_cols)

    # Reorder columns
    inv_cols_sorted = sorted([c for c in inv_df.columns if c not in ['slice_idx', 'slice_name']])
    inv_df = inv_df[['slice_idx', 'slice_name'] + inv_cols_sorted]

    # Save inverse maps CSV
    inv_df.to_csv(inv_csv_path, index=False)
    num_inv_maps = inv_df[inv_cols_sorted].notna().sum().sum() if inv_cols_sorted else 0
    logging.warning(f"Inverse map CSV saved to: {inv_csv_path}")
    logging.warning(f"  Found {num_inv_maps} inverse maps across {len(inv_cols_sorted)} transform types")

    # Diagnostic: identify mismatches between forward and inverse maps
    if num_fwd_maps != num_inv_maps:
        logging.warning(f"  MISMATCH: {num_fwd_maps} forward maps vs {num_inv_maps} inverse maps")
        # Find which (slice, transform) pairs have forward but no inverse
        common_cols = set(fwd_cols_sorted) & set(inv_cols_sorted)
        for col in common_cols:
            fwd_has = fwd_df[col].notna()
            inv_has = inv_df[col].notna()
            missing_inv = fwd_has & ~inv_has
            missing_fwd = ~fwd_has & inv_has
            if missing_inv.any():
                missing_slices = fwd_df.loc[missing_inv, 'slice_idx'].tolist()
                logging.warning(f"    Transform '{col}': slices {missing_slices} have forward map but no inverse")
            if missing_fwd.any():
                missing_slices = inv_df.loc[missing_fwd, 'slice_idx'].tolist()
                logging.warning(f"    Transform '{col}': slices {missing_slices} have inverse map but no forward")
        # Check for columns only in one dataframe
        fwd_only_cols = set(fwd_cols_sorted) - set(inv_cols_sorted)
        inv_only_cols = set(inv_cols_sorted) - set(fwd_cols_sorted)
        if fwd_only_cols:
            logging.warning(f"    Columns with forward maps only (no inverse column): {fwd_only_cols}")
        if inv_only_cols:
            logging.warning(f"    Columns with inverse maps only (no forward column): {inv_only_cols}")


# ==================================================================================
# STEP: Apply transforms to original (non-SDF) images
# ==================================================================================
# This section projects the _orig.nii.gz files (raw tissue images before SDF conversion)
# into the final registered spaces by composing and applying the forward coordinate mappings.
# The image is only interpolated once regardless of how many transforms are chained.
#
# User-configurable parameters:
#   - apply_transforms_to_orig: enable/disable this step
#   - orig_target_levels: list of target registration levels to project into
#   - keep_deformed_orig_slices: whether to keep individual deformed slice files
#   - generate_orig_stack: whether to generate a 3D stack of the deformed originals
#   - orig_interpolation: interpolation method ('linear' or 'nearest')
#   - orig_fill_value: fill value for out-of-bounds pixels
#   - save_composed_orig_mappings: whether to save the composed mapping per slice

apply_transforms_to_orig = False # and use_signed_distance_weighting_for_registration  # only makes sense when SDF was used (otherwise _orig == registered images)
orig_target_levels = [f'rigsyn_iter4_win12'] #,f'groupwise_iter{groupwise_iterations-1}']  # project into the final groupwise space; add more levels to project into multiple spaces
keep_deformed_orig_slices = False      # True: keep individual slice files, False: only keep the stack
generate_orig_stack = True             # True: generate a 3D stack NIfTI
orig_interpolation = 'linear'          # 'linear' or 'nearest'
orig_fill_value = 0                    # fill value for out-of-bounds pixels
save_composed_orig_mappings = False    # True: save the single composed mapping per slice (useful for debugging)
groupwise_first_chained_iter = 3       # must match use_deformed_source_after_iteration - 1 from groupwise registration

if apply_transforms_to_orig:
    for target_level in orig_target_levels:
        logging.warning(f"Projecting original images into '{target_level}' space")
        stack_file = apply_transforms_to_original_slices(
            output_dir=output_dir,
            subject=subject,
            all_image_fnames=all_image_fnames,
            forward_maps_csv=fwd_csv_path,
            target_level=target_level,
            zfill_num=zfill_num,
            max_workers=max_workers,
            interpolation=orig_interpolation,
            fill_value=orig_fill_value,
            keep_deformed_slices=keep_deformed_orig_slices,
            generate_stack=generate_orig_stack,
            save_composed_mappings=save_composed_orig_mappings,
            groupwise_first_chained_iter=groupwise_first_chained_iter,
            orig_suffix='_coreg0nl_ants-def0',
            voxel_res=voxel_res,
            missing_idxs_to_fill=missing_idxs_to_fill,
        )
        if stack_file:
            logging.warning(f"Original image stack in '{target_level}' space: {stack_file}")


# ==================================================================================
# STEP: Export the final densities as OME-Zarr (NGFF 0.6, tck_view format), see export_ome_zarr.py
# ==================================================================================
# stack: registered grid in mm. The mri placement is made by mri_register.py (run it by hand for the level you want):
# a new affine fit for every export was not stable (one of four was turned by 180 deg, 2026-10-28), and that script
# uses a stored reference affine with a stability check.
# needs tck_view's uv environment (~/Documents/code/tck_view); a failure is logged and does not stop the run
import subprocess
final_level = f'{input_source_file_tag}_groupwise_iter{groupwise_iterations-1}'
export_zarr = os.environ.get('SLICEREG_EXPORT_ZARR', '1') == '1' and not smoke_test
if export_zarr:
    for placement in ['stack']:
        zarr_out = f'{output_dir}{subject}_{final_level}_density_{placement}.ome.zarr'
        if os.path.isdir(zarr_out):
            logging.warning(f'OME-Zarr exists, skipping: {zarr_out}')
            continue
        cmd = [sys.executable, os.path.join(script_dir, 'export_ome_zarr.py'), '--run-dir', output_dir,
               '--rescale', str(rescale), '--level', final_level, '--placement', placement, '--out', zarr_out,
               '--direction', str(int(os.environ.get('SLICEREG_MRI_DIRECTION', 1))),
               '--missing', ','.join(map(str, missing_idxs_to_fill or []))]
        try:
            subprocess.run(cmd, check=True)
            logging.warning(f'OME-Zarr written: {zarr_out}')
        except Exception as e:
            logging.warning(f'OME-Zarr export failed for {placement}: {e}')


# ==================================================================================
# STEP: QC report (qc_report.py): <output_dir>/QC/index.html; mri_register.py adds its results to it
# ==================================================================================
if os.environ.get('SLICEREG_QC', '1') == '1':
    try:
        subprocess.run([sys.executable, os.path.join(script_dir, 'qc_report.py'), '--run-dir', output_dir,
                        '--rescale', str(rescale), '--subject', subject,
                        '--missing', ','.join(map(str, missing_idxs_to_fill or []))], check=True)
        logging.warning(f'QC report written: {output_dir}QC/index.html')
    except Exception as e:
        logging.warning(f'QC report failed: {e}')
