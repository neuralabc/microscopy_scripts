
import os
import time
import shutil
import logging
import sys
from datetime import datetime
import nighres
import numpy
import nibabel
import glob
from PIL import Image
import pandas as pd
from scipy.signal import convolve2d
import math
from nighres.io import load_volume, save_volume




# code by @pilou, using nighres; adapted, modularized, extended, and parallelized registrations by @csteele
## Potential list of todo's
# TODO: keep track of MI fits, only update template when MI indicates it is necessary? 
# TODO: additional weight of registrations by MI to downweight slices that are much different (much more processing)
# TODO: potentially incorporate mesh creation to either identify mask (limiting registration)
#       potentially included as a distance map in some way to weight boundary?

# file parameters
subject = 'zefir'


zfill_num = 4
per_slice_template = True #use a median of the slice and adjacent slices to create a slice-specific template for anchoring the registration
# rescale=10 #larger scale means that you have to change the scaling_factor
rescale=40
downsample_parallel = False #True means that we invoke Parallel, but can be much faster when set to False since it skips the Parallel overhead
# max_workers = 50 #number of parallel workers to run for registration -> registration is slow but not CPU bound on an HPC (192 cores could take ??)
max_workers = 10 #number of parallel workers to run for registration -> registration is slow but not CPU bound on an HPC (192 cores could take ??)


# output_dir = '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/slice_reg_perSliceTemplate_image_weights_all_tmp/'
output_dir = f'/tmp/slice_reg_perSliceTemplate_image_weights_dwnsmple_parallel_{rescale}/'

# registration parameters
# scaling_factor = 32 #32 or 64 for full?
scaling_factor = 8
_df = pd.read_csv('/data/data_drive/Macaque_CB/processing/results_from_cell_counts/all_TP_image_idxs_file_lookup.csv')
# _df = pd.read_csv('/data/neuralabc/neuralabc_volunteers/macaque/all_TP_image_idxs_file_lookup.csv')
all_image_fnames = list(_df['file_name'].values)
all_image_fnames = all_image_fnames[0:7] #for testing

# set missing indices, which will be iteratively filled with the mean of the neighbouring slices
# missing_idxs_to_fill = [32,59,120,160,189,228] #these are the slice indices with missing or terrible data, fill with mean of neigbours
missing_idxs_to_fill = [5]
if missing_idxs_to_fill is not None:
    if numpy.max(numpy.array(missing_idxs_to_fill)) > len(all_image_fnames): #since these are indices, will start @ 0
        raise ValueError("Missing slice indices exceed the number of images in the stack.")

# all_image_fnames = all_image_fnames[0:10] #for testing
all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below


if not os.path.exists(output_dir):
     os.makedirs(output_dir)

def generate_gaussian_weights(slice_order_idxs, gauss_std=3):
    """
    Generates Gaussian weights for the given slice indices, ensuring the weights sum to 1.
    This should be agnostic to the order in which the slice_order_indices are input, but this order
    should be consistent with the input slices. 0 must be the first element of slice_order_idxs, as 
    this indicates the position of the template and will receive the peak of the gaussian.
    
    Parameters:
    - slice_order_idxs: list of slice indices to generate weights for.
    - gauss_std: Standard deviation of the Gaussian distribution, controls the spread of the weights.
    
    Returns:
    - out_weights: Array of Gaussian weights corresponding to the input slice indices, summing to 1.
    """
    import numpy as np
    from scipy import signal

    # Ensure slice_order_idxs is a numpy array
    slice_order_idxs = np.array(slice_order_idxs)
    
    # Insert 0 into the beginning of the slice_order_idxs if it is not already there
    if 0 not in slice_order_idxs:
        slice_order_idxs = np.insert(slice_order_idxs,0,0) #insert 0 at the beginning
    if slice_order_idxs[0] != 0:
        print("0 must be the first element of slice_order_idxs")
        print("FIX THIS")
        return "0 must be the first emelement of slice_order_idxs"
    elif np.sum(slice_order_idxs==0)>1:
        print("There are multiple 0s in slice_order_idxs")
        print("FIX THIS")
        return "There are multiple 0s in slice_order_idxs"
    
    # Define the range of indices to cover both positive and negative slices symmetrically
    max_idx = np.max(np.abs(slice_order_idxs))
    num_vals = max_idx * 2 + 1  # Total number of values in the symmetric Gaussian
    
    # Generate a symmetric Gaussian, centered at 0
    gaussian_window = signal.windows.gaussian(num_vals, std=gauss_std)
    
    # Extract the weights corresponding to the absolute slice indices
    out_weights = np.zeros(slice_order_idxs.shape)
    
    for i, slice_idx in enumerate(slice_order_idxs):
        # Use the absolute value of the slice index to get the corresponding weight
        out_weights[i] = gaussian_window[max_idx + slice_idx]
    
    # Normalize the weights to sum to 1
    return out_weights / out_weights.sum()

def coreg_single_slice_orig(idx, output_dir, subject, img, all_image_names, template, 
                       target_slice_offset_list=[-1, -2, -3], zfill_num=4, 
                       input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl',
                       run_syn=True, run_rigid=True, previous_target_tag=None, 
                       scaling_factor=64, image_weights=None, per_slice_template=False):
    """
    Register a single slice and its neighboring slices based on offsets.
    """

    # logging.warning('----------------------')

    # logging.warning(input_source_file_tag)
    # logging.warning(template)
    # logging.warning(input_source_file_tag)
    # logging.warning(input_source_file_tag)

    img_basename = os.path.basename(img).split('.')[0]
    if previous_target_tag is not None:
        previous_tail = f'_{previous_target_tag}_ants-def0.nii.gz' #if we want to use the previous iteration rather than building from scratch every time (useful for windowing)
    else:
        previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'
    # previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'
    
    nifti = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_basename}{previous_tail}"
    sources = [nifti]
    image_weights_ordered = [image_weights[0]]
    # Assign the correct template for this slice
    
    ## TODO: remove per_slice_template from function call and just use the template directly
    if type(template) is list:
        targets = [template[idx]]
    else:
        targets = [template]

    # if per_slice_template:
    #     targets = [template[idx]]
    # else:
    #     targets = [template]
    
    # Determine the sources and targets based on `target_slice_offset_list`
    for idx2, slice_offset in enumerate(target_slice_offset_list):
        if slice_offset < 0 and idx >= abs(slice_offset):        
            prev_nifti = f"{output_dir}{subject}_{str(idx + slice_offset).zfill(zfill_num)}_{all_image_names[idx + slice_offset]}{previous_tail}"
            sources.append(nifti)
            targets.append(prev_nifti)
            image_weights_ordered.append(image_weights[idx2 + 1])
        elif slice_offset > 0 and idx < len(all_image_names) - slice_offset:
            next_nifti = f"{output_dir}{subject}_{str(idx + slice_offset).zfill(zfill_num)}_{all_image_names[idx + slice_offset]}{previous_tail}"
            sources.append(nifti)
            targets.append(next_nifti)
            image_weights_ordered.append(image_weights[idx2 + 1])
            
    # logging.warning('Targets:')
    # for t in targets:
    #     try:
    #         logging.warning(f'\t{t.split("/")[-1]}')
    #     except:
    #         logging.warning(t)    
    # logging.warning('Sources:')
    # for s in sources:
    #     try:
    #         logging.warning(f'\t{s.split("/")[-1]}')
    #     except:
    #         logging.warning(s)
    # logging.warning(image_weights_ordered)

    output = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_basename}_{reg_level_tag}"
    coreg_output = nighres.registration.embedded_antspy_2d_multi(
        source_images=sources,
        target_images=targets,
        image_weights=image_weights_ordered,
        run_rigid=run_rigid,
        rigid_iterations=1000,
        run_affine=False,
        run_syn=run_syn,
        coarse_iterations=2000,
        medium_iterations=1000, 
        fine_iterations=200,
        scaling_factor=scaling_factor,
        cost_function='MutualInformation',
        interpolation='Linear',
        regularization='High',
        convergence=1e-6,
        mask_zero=False,
        ignore_affine=True, 
        ignore_orient=True, 
        ignore_res=True,
        save_data=True, 
        overwrite=False,
        file_name=output
    )
    
    # Clean up unnecessary files
    def_files = glob.glob(f'{output}_ants-def*')
    for f in def_files:
        if 'def0' not in f:
            os.remove(f)
            time.sleep(0.5)
    logging.warning(f"\t\tRegistration completed for slice {idx}.")


from concurrent.futures import ProcessPoolExecutor, as_completed

def run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction=None, max_workers=3, 
                                  target_slice_offset_list=[-1,-2,-3], zfill_num=4, input_source_file_tag='coreg0nl', 
                                  reg_level_tag='coreg1nl', run_syn=True, run_rigid=True, previous_target_tag=None, 
                                  scaling_factor=64, image_weights=None, per_slice_template=False):
    # reverse direction uses the same function, but target_slice_offset list is negative to ensure proper lookup.
    # the actual order of registrations is the same 0 -> n:
    #   for the reverse direction, we register to slices after the current slice (named by how you would walk through the loop of the stack)
    #   for the forward direction, we register to slices before the current slice

    if direction not in ['forward', 'reverse']:
        raise ValueError("Invalid direction. Must be either 'forward' or 'reverse'.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        if direction == 'forward':
            for idx, img in enumerate(all_image_fnames):
                futures.append(
                    executor.submit(coreg_single_slice_orig, idx, output_dir, subject, img, all_image_names, template, 
                                    target_slice_offset_list=target_slice_offset_list, zfill_num=zfill_num, 
                                    input_source_file_tag=input_source_file_tag, reg_level_tag=reg_level_tag,
                                    run_syn=run_syn, run_rigid=run_rigid, previous_target_tag=previous_target_tag,
                                    scaling_factor=scaling_factor, image_weights=image_weights, per_slice_template=per_slice_template)
                )
        elif direction == 'reverse':
            for idx, img in enumerate(all_image_fnames):
                futures.append(
                    executor.submit(coreg_single_slice_orig, idx, output_dir, subject, img, all_image_names, template, 
                                    target_slice_offset_list=target_slice_offset_list, zfill_num=zfill_num, 
                                    input_source_file_tag=input_source_file_tag, reg_level_tag=reg_level_tag,
                                    run_syn=run_syn, run_rigid=run_rigid, previous_target_tag=previous_target_tag,
                                    scaling_factor=scaling_factor, image_weights=image_weights, per_slice_template=per_slice_template)
                )
        for future in as_completed(futures):
            try:
                future.result()
                # logging.warning("Registration completed for one slice.")
            except Exception as e:
                logging.error(f"Registration failed with error: {e}")



def compute_intermediate_non_linear_slice(pre_img, post_img):
    """
    Computes an intermediate slice by averaging both rigid and non-linear transformations.
    Images must already fit within the same matrix (i.e., have the same dimensions).

    Parameters:
    pre_img (str): Filename of the pre-slice image.
    post_img (str): Filename of the post-slice image.

    Returns:
    intermediate_img_np (numpy.ndarray): Intermediate slice computed between pre_img and post_img as a NumPy array.
    """
    import tempfile
    import ants

    # Load images using ANTs
    pre_ants = ants.image_read(pre_img)
    post_ants = ants.image_read(post_img)

    # Step 1: Perform rigid registration from pre to post slice and post to pre slice
    pre_to_post_rigid = ants.registration(fixed=post_ants, moving=pre_ants, type_of_transform='Rigid')
    post_to_pre_rigid = ants.registration(fixed=pre_ants, moving=post_ants, type_of_transform='Rigid')

    # Step 2: Apply the rigid transformation to each image for initial alignment
    pre_aligned = ants.apply_transforms(fixed=post_ants, moving=pre_ants, transformlist=pre_to_post_rigid['fwdtransforms'])
    post_aligned = ants.apply_transforms(fixed=pre_ants, moving=post_ants, transformlist=post_to_pre_rigid['fwdtransforms'])


    # # Step 3: Perform non-linear registration on the rigidly aligned images
    pre_to_post_nonlin = ants.registration(fixed=post_ants, moving=pre_aligned, type_of_transform='SyN')
    post_to_pre_nonlin = ants.registration(fixed=pre_ants, moving=post_aligned, type_of_transform='SyN')

    # # Step 3: Perform non-linear registration on the rigidly aligned images
    # pre_to_post_nonlin = ants.registration(fixed=pre_aligned, moving=post_aligned, type_of_transform='SyN')
    # post_to_pre_nonlin = ants.registration(fixed=post_aligned, moving=pre_aligned, type_of_transform='SyN')

    # Step 4: Load the non-linear deformation fields as images
    pre_to_post_field = ants.image_read(pre_to_post_nonlin['fwdtransforms'][0])
    post_to_pre_field = ants.image_read(post_to_pre_nonlin['fwdtransforms'][0])


    # Step 5: Convert the deformation fields to NumPy arrays and average them
    avg_field_data = (pre_to_post_field.numpy() + post_to_pre_field.numpy()) / 2
    avg_field = ants.from_numpy(avg_field_data, spacing=pre_to_post_field.spacing+(1.0,)) #need a 3rd dimension for spacing
    
    #we need to have the transform as a file, so we create a temp version here
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as temp_file:
        avg_field_path = temp_file.name
        ants.image_write(avg_field, avg_field_path)

        ## apply within the with statement to use the file prior to deletion (default is non- persistence)
        # Step 6: Apply the averaged non-linear deformation field to the rigidly aligned pre-image
        intermediate_img = ants.apply_transforms(fixed=post_ants, moving=pre_aligned, transformlist=[avg_field_path])

    # Convert to NumPy array
    # TODO: FIGURE OUT HOW TO IDENTIFY THE ROTATION ISSUE and SOLVE directly rather than hacking to this rotation (b/c this is relative to the input orientation...)
    intermediate_img_np = numpy.rot90(intermediate_img.numpy(),k=2) #rotate 180 degrees, since we are in different spaces


    return intermediate_img_np



def generate_missing_slices(missing_fnames_pre,missing_fnames_post,method='intermediate_nonlin_mean'):
    '''
    Generate missing slices by averaging the neighbouring slices in multiple ways
    method = 'mean' : average of two neighbouring slices
    method = 'intermediate_nonlin_mean': applying a non-linear transformation to the neighbouring slices, averaging the deformation, then applying to the pre-slice

    '''
    if method == 'mean':
        # Load pre and post images
        pre_slices = []
        for img_fname in missing_fnames_pre:
            img_data = nighres.io.load_volume(img_fname).get_fdata()
            pre_slices.append(img_data)
        pre_slices = numpy.stack(pre_slices, axis=-1)

        post_slices = []
        for img_fname in missing_fnames_post:
            img_data = nighres.io.load_volume(img_fname).get_fdata()
            post_slices.append(img_data)
        post_slices = numpy.stack(post_slices, axis=-1)
    
        missing_slices_interpolated = .5*(pre_slices+post_slices)
    
    elif method == 'intermediate_nonlin_mean':
        missing_slices_interpolated = []
        for idx,img_fname in enumerate(missing_fnames_pre):
            img_fname_pre = missing_fnames_pre[idx]
            img_fname_post = missing_fnames_post[idx]
            missing_slices_interpolated.append(compute_intermediate_non_linear_slice(img_fname_pre,img_fname_post))
        missing_slices_interpolated= numpy.stack(missing_slices_interpolated, axis=-1)
    else:
        missing_slices_interpolated = None
    return missing_slices_interpolated

def generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=4,reg_level_tag='coreg12nl',
                                per_slice_template=False,missing_idxs_to_fill=None):
    
    #we can also output a per_slice_template based on the median of the current and neighbouring slices
    stack = []
    stack_tail = f'_{reg_level_tag}_stack.nii.gz'
    img_stack = output_dir+subject+stack_tail
    template_tail = f'_{reg_level_tag}_template.nii.gz'
    template = output_dir+subject+template_tail

    template_list = []

    img_tail = f'_{reg_level_tag}_ants-def0.nii.gz'

    # if (os.path.isfile(img_stack)):
    if False:
        print('Stacking was already completed for this level: {}'.format(template))
    else:
        #this can only handle if there is a single missing slice between two good slices
        #and that they are not at the start or end of the stack (or it will crash)
        for idx,img_name in enumerate(all_image_fnames):
            img_name = os.path.basename(img_name).split('.')[0]
            reg = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+img_tail
            stack.append(nighres.io.load_volume(reg).get_fdata())

        img = numpy.stack(stack,axis=-1)

        #now we fill any missing data with the mean of the neighbouring slices
        if missing_idxs_to_fill is not None and len(missing_idxs_to_fill)>0:

            #we generate the filenames and then pass them to a helper function to generate the missing slices (as an array)
            missing_idxs_to_fill.sort() #sort it
            missing_idxs_pre = numpy.array(missing_idxs_to_fill)-1
            missing_idxs_post = numpy.array(missing_idxs_to_fill)+1
                        
            for idx,img_idx in enumerate(missing_idxs_pre):
                if idx ==0:
                    missing_fnames_pre = []
                
                img_name = all_image_fnames[img_idx]
                img_name = os.path.basename(img_name).split('.')[0]
                reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
                missing_fnames_pre.append(reg)
                
            for idx,img_idx in enumerate(missing_idxs_post):
                if idx == 0:
                    missing_fnames_post = []
                img_name = all_image_fnames[img_idx]
                img_name = os.path.basename(img_name).split('.')[0]
                reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
                missing_fnames_post.append(reg)

            missing_slices_interpolated = generate_missing_slices(missing_fnames_pre,missing_fnames_post)

            ## OLD WAY, restricted to just the mean
            # missing_idxs_to_fill.sort() #sort it
            # missing_slices_interpolated = []
            # missing_idxs_pre = numpy.array(missing_idxs_to_fill)-1
            # missing_idxs_post = numpy.array(missing_idxs_to_fill)+1            
            # for idx,img_idx in enumerate(missing_idxs_pre):
            #     img_name = all_image_fnames[img_idx]
            #     img_name = os.path.basename(img_name).split('.')[0]
            #     reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
            #     _t = nighres.io.load_volume(reg).get_fdata()
            #     if idx==0:
            #         pre_d = numpy.zeros(_t.shape+(len(missing_idxs_to_fill),))
            #     pre_d[...,idx] = _t

            # for idx,img_idx in enumerate(missing_idxs_post):
            #     img_name = all_image_fnames[img_idx]
            #     img_name = os.path.basename(img_name).split('.')[0]
            #     reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
            #     _t = nighres.io.load_volume(reg).get_fdata()
            #     if idx==0:
            #         post_d = numpy.zeros(_t.shape+(len(missing_idxs_to_fill),))
            #     post_d[...,idx] = _t
            
            # missing_slices_interpolated = .5*(pre_d+post_d)

            #now we can fill the slices with the interpolated value
            for idx,missing_idx in enumerate(missing_idxs_to_fill):
                print(idx)
                print(missing_idx)
                print(numpy.shape(missing_slices_interpolated))
                print(numpy.shape(img))
                img[...,missing_idx] = missing_slices_interpolated[...,idx]

        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(img_stack,nifti)

        if per_slice_template:
            num_slices = img.shape[-1]
            for idx,img_name in enumerate(all_image_fnames):
                img_name = os.path.basename(img_name).split('.')[0]
                slice_template_fname = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_tail
                if idx == 0: #if at the front, take the first two only
                    slice_template = numpy.median(img[...,0:2],axis=-1)
                elif idx == num_slices-1: #if at the end, take the last two only
                    slice_template = numpy.median(img[...,-2:],axis=-1)
                else: #take one on each side and the current slice
                    start = idx-1
                    stop = idx+2
                    slice_template = numpy.median(img[...,start:stop],axis=-1)

                header.set_data_shape(slice_template.shape)
                nifti = nibabel.Nifti1Image(slice_template,affine=None,header=header)
                nifti.update_header()
                save_volume(slice_template_fname,nifti)
                template_list.append(slice_template_fname)            

        img = numpy.median(img,axis=2)
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(template,nifti)
        print('Stacking: done - {}'.format(template))
    if per_slice_template:
        return template_list
    else:
        return template
    
def register_stack_to_mri(slice_stack_template, mri_template):
    # Registration of the entire 2D slice stack to the 3D MRI template.
    # TODO: check what outputs are and figure out how to get the full filename if it is not provided (think it is the nimg?)
    # TODO: may not need [], as this is an overloaded function in nighres that does this itself
    output_aligned_stack = slice_stack_template.split('.')[0] + 'aligned_to_mri.nii.gz'
    
    aligned_stack = nighres.registration.embedded_antspy(
        source_images=[slice_stack_template],
        target_images=[mri_template],
        run_rigid=True,
        run_syn=True,
        save_data=True,
        file_name=output_aligned_stack
    )
    return aligned_stack

def select_best_reg_by_MI(output_dir,subject,all_image_fnames,df_struct=None, template_tag='coreg0nl',
                          zfill_num=zfill_num,reg_level_tag1='coreg1nl', reg_level_tag2='coreg2nl',reg_output_tag='coreg12nl',per_slice_template=False,
                          overwrite=True):
    '''
    Use MI to determine best registration (forwards or backwards) and select going forward
    reg_output_tag identifies the best registration outputs
    '''
    template_tail = f'_{template_tag}_template.nii.gz'
    out_tail = f'_{reg_output_tag}'
    tag1_tail = f'_{reg_level_tag1}'
    tag2_tail = f'_{reg_level_tag2}'
    
    for idx,img_name in enumerate(all_image_fnames):
        img_name = os.path.basename(img_name).split('.')[0]
        if idx == 0: #create lists for mutual information comparisons
            mi1c_l = []
            mi2c_l = []
            img_name_l = []

        if not per_slice_template:
            template = output_dir+subject+template_tail #we use the generally defined template
        output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+out_tail+'_ants-def0.nii.gz'
        if (not os.path.isfile(output)) or overwrite:
            if per_slice_template: #or,we use individual templates
                template = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_tail
            slice1 = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag1_tail+'_ants-def0.nii.gz'
            slice2 = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag2_tail+'_ants-def0.nii.gz'
        
            curr1 = nighres.io.load_volume(slice1).get_fdata()
            curr2 = nighres.io.load_volume(slice2).get_fdata()
            curr = nighres.io.load_volume(template).get_fdata()
            
            p1,v1 = numpy.histogram(curr1.flatten(), bins=100, density=True)
            p2,v2 = numpy.histogram(curr2.flatten(), bins=100, density=True)
            pc,vc = numpy.histogram(curr.flatten(), bins=100, density=True)

            # normalize histograms to 1
            p1 = p1/numpy.sum(p1)
            p2 = p2/numpy.sum(p2)
            pc = pc/numpy.sum(pc)
            
            p1c,v1,vc = numpy.histogram2d(curr1.flatten(), curr.flatten(), bins=100, density=True)
            p2c,v2,vc = numpy.histogram2d(curr2.flatten(), curr.flatten(), bins=100, density=True)
        
            # normalize joint histograms to 1
            p1c = p1c / numpy.sum(p1c)
            p2c = p2c / numpy.sum(p2c)
            
            p1pc = numpy.outer(p1,pc)
            p2pc = numpy.outer(p2,pc)
                
            mi1c = numpy.sum(p1c*numpy.log(p1c/(p1pc),where=(p1c*p1pc>0)))
            mi2c = numpy.sum(p2c*numpy.log(p2c/(p2pc),where=(p2c*p2pc>0)))
        
            print("MI: "+str(mi1c)+", "+str(mi2c))
            mi1c_l.append(mi1c)
            mi2c_l.append(mi2c)
            img_name_l.append(img_name)

            # copy the best result
            mapping= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+out_tail+'_ants-map.nii.gz'
            inverse= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+out_tail+'_ants-invmap.nii.gz'
            if (mi1c>mi2c): 
                mapping1= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag1_tail+'_ants-map.nii.gz'
                inverse1= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag1_tail+'_ants-invmap.nii.gz'
                shutil.copyfile(mapping1, mapping)
                shutil.copyfile(inverse1, inverse)
                shutil.copyfile(slice1, output)
            else:
                mapping2= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag2_tail+'_ants-map.nii.gz'
                inverse2= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag2_tail+'_ants-invmap.nii.gz'
                shutil.copyfile(mapping2, mapping)
                shutil.copyfile(inverse2, inverse)
                shutil.copyfile(slice2, output)

            # cleanup files, removing old mappings that are no longer needed
            map_files = glob.glob(output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+"*"+'_ants-*map.nii.gz')
            for f in map_files:
                if out_tail in f:
                    pass
                else:
                    os.remove(f)
                    time.sleep(.5)

            os.remove(slice1)
            time.sleep(.5)
            os.remove(slice2)
            time.sleep(.5)
    if df_struct is not None: #we dump out lists into the dataframe-like structure to keep track of MI values 
        df_struct['MI1c'+out_tail] = mi1c_l
        df_struct['MI2c'+out_tail] = mi2c_l
        df_struct['img_name'+out_tail] = img_name_l
        return df_struct
    else:
        return None


def downsample_block(block):
    """Helper function to compute the sum of a block."""
    return block.sum()


def downsample_image_ORIG(image, rescale, prop_pad=.2):
    """
    Downsamples a 2D image by summing over rescale x rescale blocks. Pad by prop_pad before downsampling to ensure all data is within the final registered image(s)
    
    Parameters:
    - image (numpy.ndarray): Input 2D image to downsample.
    - rescale (int): Factor by which to downsample.
    - prop_pad (float): Proportion of padding to add to each border of the image before downsampling

    Returns:
    - numpy.ndarray: Downsampled image with summed values in blocks.
    """
    from skimage.measure import block_reduce
    size0 = image.shape[0]
    size1 = image.shape[1]
    pad0 = math.ceil(size0+size0*prop_pad)
    pad1 = math.ceil(size1+size1*prop_pad)

    # Ensure image dimensions are compatible with rescale factor
    pad_width = ((rescale-pad0%rescale, rescale - pad0%rescale), 
                 (rescale-pad1%rescale, rescale - pad1%rescale))
    # logging.warning(pad_width)
    padded_image = numpy.pad(image, pad_width=pad_width, mode='edge')
    
    # Downsample by block summing
    downsampled_image = block_reduce(padded_image, block_size=(rescale, rescale), func=numpy.sum)
    return downsampled_image

def downsample_image(image, rescale, prop_pad=0.2):
    """
    Downsamples a 2D image by summing over rescale x rescale blocks. Pad by prop_pad 
    before downsampling to ensure all data is within the final registered image(s).
    
    Parameters:
    - image (numpy.ndarray): Input 2D image to downsample.
    - rescale (int): Factor by which to downsample.
    - prop_pad (float): Proportion of padding to add to each border of the image before downsampling.

    Returns:
    - numpy.ndarray: Downsampled image with summed values in blocks.
    """
    from skimage.measure import block_reduce

    # Original dimensions
    size0, size1 = image.shape

    # Calculate padding based on proportion
    pad0 = math.ceil(size0 * prop_pad)
    pad1 = math.ceil(size1 * prop_pad)

    # Pad each side equally
    total_pad_width = ((pad0, pad0), (pad1, pad1))
    padded_image = numpy.pad(image, pad_width=total_pad_width, mode='constant', constant_values=0)

    # Adjust padding to ensure divisibility by rescale
    padded_size0, padded_size1 = padded_image.shape
    extra_pad_width = ((rescale - padded_size0 % rescale) % rescale,
                       (rescale - padded_size1 % rescale) % rescale)

    padded_image = numpy.pad(padded_image, 
                          ((0, extra_pad_width[0]), (0, extra_pad_width[1])),
                          mode='constant', constant_values=0)

    # Downsample by block summing
    downsampled_image = block_reduce(padded_image, block_size=(rescale, rescale), func=numpy.sum)
    
    return downsampled_image

def downsample_image_parallel(image, rescale, n_jobs=-1):
    """
    Downsamples a 2D image by summing over rescale x rescale blocks in parallel.
    
    Parameters:
    - image (numpy.ndarray): Input 2D image to downsample.
    - rescale (int): Factor by which to downsample.
    - n_jobs (int): Number of CPU cores to use; -1 uses all available cores.
    
    Returns:
    - numpy.ndarray: Downsampled image with summed values in blocks.
    """
    from skimage.util import view_as_blocks
    from joblib import Parallel, delayed
    np = numpy

    # Pad image to match rescale size
    pad_width = ((0, rescale - image.shape[0] % rescale), 
                 (0, rescale - image.shape[1] % rescale))
    padded_image = np.pad(image, pad_width=pad_width, mode='edge')

    # View as blocks of shape (rescale, rescale)
    blocks = view_as_blocks(padded_image, block_shape=(rescale, rescale))
    # Flatten the blocks along the first two dimensions for parallel processing
    flat_blocks = blocks.reshape(-1, rescale, rescale)

    # Process each block in parallel, summing within each block
    downsampled_values = Parallel(n_jobs=n_jobs)(
        delayed(downsample_block)(block) for block in flat_blocks
    )

    # Reshape results back into the downsampled image shape
    downsampled_image = np.array(downsampled_values).reshape(blocks.shape[:2])
    
    return downsampled_image
            
def create_affine(shape):
    """
    Creates an affine transformation matrix centered on the image.
    
    Parameters:
    - shape (tuple): Shape of the downsampled image.
    
    Returns:
    - numpy.ndarray: 4x4 affine matrix.
    """
    affine = numpy.eye(4)
    affine[0, 3] = -shape[0] / 2.0
    affine[1, 3] = -shape[1] / 2.0
    return affine


## output logger
class StreamToLogger:
    """Redirect `print` statements to the logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, message):
        if message.strip():  # Log only non-empty messages
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass  # Required for file-like objects, no action needed here

def setup_logging(dataset_name, out_dir):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Format log file name based on the dataset and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(out_dir, f"{dataset_name}_log_{timestamp}.log")
    
    # Configure the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all messages
    
    # Create a file handler that writes all messages to the log file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add a console handler to output higher-priority messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect `print` statements to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    
    # Log the start of processing
    logger.info(f"Logging initialized for dataset '{dataset_name}'")
    
    return logger


# start our logger, which will capture all the print statements
script_name = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = setup_logging(script_name, output_dir)

print(f"Output directory: {output_dir}")
exit
# 0. Convert to nifti
print('0. Converting images to .nii.gz')
logger.warning('0. Converting images to .nii.gz') #use warnings so that we can see progress on command line as well as in the log file
for idx,img_orig in enumerate(all_image_fnames):
    img = os.path.basename(img_orig).split('.')[0] 
    output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'
    
    if (os.path.isfile(output)):
        print('\t - already done, using existing image')
        nifti = output
    else:
        print('\t - image '+str(img_orig))
        # get the TIFF image
        slice_name = str(img_orig)
        if os.path.isfile(slice_name):
            slice_img = Image.open(slice_name)
            
            slice_img = numpy.array(slice_img)
            
            # crop: use various options, padding to ensure multiple of rescale
            # image = slice_img
            slice_li = numpy.pad(slice_img,pad_width=((0,rescale),(0,rescale)),mode='edge')
            
            
            if downsample_parallel:
                slice_img = downsample_image_parallel(slice_img, rescale, n_jobs=-1)
            else:
                slice_img = downsample_image(slice_img, rescale)

            ## original approach, below, was v. slow
            ## alternative using 2d convolution to preserve cell counts (meaning is still the same here)
            # kernel = numpy.ones((rescale,rescale)) #2d convolution kernel, all 1s
            # slice_img = convolve2d(image,kernel,mode='full')[::rescale,::rescale] #can divide by rescale if we want the mean, otherwise sum is good (total cell count)

            #exceptions that need fixing, since rigid reg does not seem to address big flips
            if ('TP1' in img_orig) or ('/testProject/' in img_orig): #we have files named the same within the subdirs, so we must specify specifically the subdir (different on local vs server) 
                if 'Image_11_-_20x_01_cellCount' in img_orig:
                    slice_img = numpy.flip(slice_img,axis=0) #flip x
            
            header = nibabel.Nifti1Header()
            header.set_data_shape(slice_img.shape)
            
            affine = create_affine(slice_img.shape)
                  
            nifti = nibabel.Nifti1Image(slice_img,affine=affine,header=header)
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
    shape = nighres.io.load_volume(nifti).header.get_data_shape()
    
    if shape[0]*shape[1]>size:
        size = shape[0]*shape[1]
        largest = idx
        
template = output_dir+subject+'_'+str(largest).zfill(zfill_num)+'_'+os.path.basename(all_image_fnames[largest]).split('.')[0]+'.nii.gz'    

print(f"\tUsing the following image as the template for size: {template}")

print('2. Bring all image slices into same place as our 2d template with an initial translation registration')
logger.warning('2. Bring all image slices into same place as our 2d template with an initial translation registration')
# initial step to bring all images into the same space of our 2d template
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]
    nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'

    sources = [nifti]
    targets = [template]
        
    output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'_coreg0nl.nii.gz'
    coreg1nl = nighres.registration.embedded_antspy_2d_multi(source_images=sources, 
                    target_images=targets,
                    run_rigid=False,
                    run_affine=False,
                    run_syn=False,
                    scaling_factor=64,
                    cost_function='MutualInformation',
                    interpolation='Linear',
                    regularization='High',
                    convergence=1e-6,
                    mask_zero=False,
                    ignore_affine=False, ignore_orient=False, ignore_res=False,
                    save_data=True, overwrite=False,
                    file_name=output)

template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=4,reg_level_tag='coreg0nl',
                                       missing_idxs_to_fill=None)
## ****************************** Iteration 1
# in all cases, we go:
#   - forwards
#   - backwards
#   - select the best registration
#   - generate a template, using a per-slice template helps quite a bit
#   - window in front and behind, forwards
#   - window in front and behind, backwards
#   - select the best registration with MI
#   - generate a template
#   - delete unecessary files (in progress...)

print('3. Begin STAGE1 registration iterations - Rigid + Syn')
logger.warning('3. Begin STAGE1 registration iterations - Rigid + Syn')

# STEP 1: Rigid + Syn
num_reg_iterations = 10
run_rigid = True
run_syn = True
template_tag = 'coreg0nl' #initial template tag, which we update with each loop
MI_df_struct = {}

for iter in range(num_reg_iterations): 
    
    #here we always go back to the original coreg0 images, we are basically just refning our target template(s)
    
    iter_tag = f"_rigsyn_{iter}"
    print(f'\t iteration tag: {iter_tag}')
    logger.warning('****************************************************************************')
    logger.warning(f'\titeration {iter_tag}')
    logger.warning('****************************************************************************')
    
    if (iter == 0):
        first_run_slice_template = False #skip using the per slice template on the first 2 reg steps below (up until the next template is created)
    else:
        first_run_slice_template = per_slice_template

    slice_offset_list_forward = [-1,-2,-3]
    slice_offset_list_reverse = [1,2,3]
    image_weights = generate_gaussian_weights([0,1,2,3]) #symmetric gaussian, so the same on both sides

    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='forward', max_workers=max_workers, 
                                 target_slice_offset_list=slice_offset_list_forward, 
                zfill_num=zfill_num, input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl'+iter_tag,
                image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='reverse', max_workers=max_workers, 
                                 target_slice_offset_list=slice_offset_list_reverse, 
                        zfill_num=zfill_num, input_source_file_tag='coreg0nl', reg_level_tag='coreg2nl'+iter_tag,
                        image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)

    # print(iter)
    # print(template_tag)
    # print(first_run_slice_template)
    logging.warning('\t\tSelecting best registration by MI')
    select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
                        zfill_num=zfill_num,reg_level_tag1='coreg1nl'+iter_tag, reg_level_tag2='coreg2nl'+iter_tag,
                        reg_output_tag='coreg12nl'+iter_tag,per_slice_template=first_run_slice_template,df_struct=MI_df_struct) #use the per slice templates after the first round, if requested
    if MI_df_struct is not None:
        pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
    
    logging.warning('\t\tGenerating new template')
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                        zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                        missing_idxs_to_fill=missing_idxs_to_fill)
   

    
    ## TODO: insert in here the code to register the stack to the MRI template and then update the tag references as necessary
    # if iter > 0: #we do not do this on the first iteration
        # MRI_reg_output = register_stack_to_mri(slice_stack_template, mri_template)

    template_tag = 'coreg12nl'+iter_tag
    
    slice_offset_list_forward = [-3,-2,-1,1,2] #weighted back, but also forward
    slice_offset_list_reverse = [-2,-1,1,2,3] #weighted forward, but also back
    image_weights = generate_gaussian_weights([0,-3,-2,-1,1,2]) #symmetric gaussian, so the same on both sides

    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='forward', max_workers=max_workers,
                                 target_slice_offset_list=slice_offset_list_forward, 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', 
                    previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win1'+iter_tag,
                    image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    image_weights = generate_gaussian_weights([0,-2,-1,1,2,3])
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='reverse', max_workers=max_workers,
                                 target_slice_offset_list=slice_offset_list_reverse, 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', 
                    previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win2'+iter_tag,
                    image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    logging.warning('\t\tSelecting best registration by MI')                                     
    select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
                        zfill_num=zfill_num,reg_level_tag1='coreg12nl_win1'+iter_tag, reg_level_tag2='coreg12nl_win2'+iter_tag,
                        reg_output_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,df_struct=MI_df_struct)
    if MI_df_struct is not None:
        pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
    
    logging.warning('\t\tGenerating new template')
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                        zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                        missing_idxs_to_fill=missing_idxs_to_fill)
    template_tag = 'coreg12nl_win12'+iter_tag
    
final_reg_level_tag = 'coreg12nl_win12'+iter_tag
step1_iter_tag = iter_tag

## TODO: ADAPT AFTER ABOVE WORKING

# # # STEP 2: Syn only
print('4. Begin STAGE2 registration iterations - Syn')
logger.warning('4. Begin STAGE2 registration iterations - Syn')
run_rigid = False
run_syn = True
num_syn_reg_iterations = 5
for iter in range(num_syn_reg_iterations):
    #for the nonlinear step, we base our registrations on the previous ones instead of going back to the original images, starting with the previous step and 
    # then using the output from each successive step
    iter_tag = f"{step1_iter_tag}_syn_{iter}"
    print(f'\t iteration tag: {iter_tag}')
    logger.warning(f'\titeration {iter_tag}')

    slice_offset_list_forward = [-1,-2,-3] #weighted back
    slice_offset_list_reverse = [1,2,3] #weighted forward
    image_weights = generate_gaussian_weights([0,1,2,3])

    # coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offset_list=slice_offset_list_forward, 
    #                 zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg1nl'+iter_tag,image_weights=image_weights,
    #                 run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    
    # coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offset_list=slice_offset_list_reverse, 
    #                         zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg2nl'+iter_tag,image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor) 
    
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='forward', max_workers=max_workers, 
                                 target_slice_offset_list=slice_offset_list_forward, 
                zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg1nl'+iter_tag,
                image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='reverse', max_workers=max_workers, 
                                 target_slice_offset_list=slice_offset_list_reverse, 
                        zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg2nl'+iter_tag,
                        image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    
    logging.warning('\t\tSelecting best registration by MI')    
    select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
                        zfill_num=zfill_num,reg_level_tag1='coreg1nl'+iter_tag, reg_level_tag2='coreg2nl'+iter_tag,
                        reg_output_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,df_struct=MI_df_struct)
    if MI_df_struct is not None:
        pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
    
    logging.warning('\t\tGenerating new template')
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                        zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                        missing_idxs_to_fill=missing_idxs_to_fill)
    template_tag = 'coreg12nl'+iter_tag
    
    ## This is, in practice, completely unecessary - staged for removal
    # slice_offset_list_forward = [-3,-2,-1,1,2] #weighted back, but also forward
    # slice_offset_list_reverse = [-2,-1,1,2,3] #weighted forward, but also back
    # image_weights = generate_gaussian_weights([0,-3,-2,-1,1,2]) #symmetric gaussian, so the same on both sides

    # # coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offset_list=slice_offset_list_forward, 
    # #                 zfill_num=zfill_num, input_source_file_tag='coreg12nl'+iter_tag, 
    # #                 previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win1'+iter_tag,image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid) 
    # # image_weights = generate_gaussian_weights([0,-2,-1,1,2,3]) #symmetric gaussian, so the same on both sides
    # # coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offset_list=slice_offset_list_reverse, 
    # #                 zfill_num=zfill_num, input_source_file_tag='coreg12nl'+iter_tag, 
    # #                 previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win2'+iter_tag,image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid)
    
    # run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='forward', max_workers=max_workers,
    #                              target_slice_offset_list=slice_offset_list_forward, 
    #                 zfill_num=zfill_num, input_source_file_tag='coreg12nl'+iter_tag, 
    #                 previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win1'+iter_tag,
    #                 image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    # image_weights = generate_gaussian_weights([0,-2,-1,1,2,3])
    # run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='reverse', max_workers=max_workers,
    #                              target_slice_offset_list=slice_offset_list_reverse, 
    #                 zfill_num=zfill_num, input_source_file_tag='coreg12nl'+iter_tag, 
    #                 previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win2'+iter_tag,
    #                 image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    
    # logging.warning('\t\tSelecting best registration by MI')
    # select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
    #                     zfill_num=zfill_num,reg_level_tag1='coreg12nl_win1'+iter_tag, reg_level_tag2='coreg12nl_win2'+iter_tag,
    #                     reg_output_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,df_struct=MI_df_struct)
    # logging.warning('\t\tGenerating new template')
    # template = generate_stack_and_template(output_dir,subject,all_image_fnames,
    #                                     zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
    #                                     missing_idxs_to_fill=missing_idxs_to_fill)
    # final_reg_level_tag = 'coreg12nl_win12'+iter_tag
    # template_tag = 'coreg12nl_win12'+iter_tag

    if MI_df_struct is not None:
        pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
        # MI_df_struct.to_csv(output_dir+subject+'_MI_values.csv',index=False)
