
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
from concurrent.futures import ProcessPoolExecutor, as_completed


#v2 still not as good as the original one!


# code by @pilou, using nighres; adapted, modularized, extended, and parallelized registrations by @csteele
## Potential list of todo's
# TODO: additional weight of registrations by MI to downweight slices that are much different (much more processing)
# TODO: potentially incorporate mesh creation to either identify mask (limiting registration)
#       potentially included as a distance map in some way to weight boundary?

# file parameters
subject = 'zefir'

#TODO: potentially carry this through to the nibabel images so that the header and affines are correct
# tried this quickly and it messed everything up...
in_plane_res_x = 10 #10 microns per pixel
in_plane_res_y = 10 #10 microns per pixel
in_plane_res_z = 50 #slice thickness of 50 microns

zfill_num = 4
per_slice_template = False #use a median of the slice and adjacent slices to create a slice-specific template for anchoring the registration

use_nonlin_slice_templates = False #use interpolated slices (from registrations of neighbouring 2 slices) as templates for registration, otherwise median
                                    # nonlinear slice templates take a long time and result in very jagged registrations, but may end up being useful for bring slices that are very far out of alignment back in
slice_template_type = 'median'
if use_nonlin_slice_templates:
    slice_template_type = [slice_template_type,'nonlin']
    
# rescale=5 #larger scale means that you have to change the scaling_factor
# scaling_factor = 64 #32 or 64 for full scaling of resolutions on the registrations
rescale=20
scaling_factor=16
#based on the rescale value, we adjust our in-plane resolution
in_plane_res_x = rescale*in_plane_res_x
in_plane_res_y = rescale*in_plane_res_y

downsample_parallel = False #True means that we invoke Parallel, but can be much faster when set to False since it skips the Parallel overhead
max_workers = 100 #number of parallel workers to run for registration -> registration is slow but not CPU bound on an HPC (192 cores could take ??)
nonlin_interp_max_workers = 100 #number of workers to use for nonlinear slice interpolation when use_nonlin_slice_templates = True

# max_workers = 10 #number of parallel workers to run for registration -> registration is slow but not CPU bound on an HPC (192 cores could take ??)



output_dir = f'/tmp/slice_reg_perSliceTemplate_image_weights_dwnsmple_parallel_v2_{rescale}_casc_v2/'
_df = pd.read_csv('/data/neuralabc/neuralabc_volunteers/macaque/all_TP_image_idxs_file_lookup.csv')
missing_idxs_to_fill = [32,59,120,160,189,228] #these are the slice indices with missing or terrible data, fill with mean of neigbours
# output_dir = '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/slice_reg_perSliceTemplate_image_weights_all_tmp/'
# _df = pd.read_csv('/data/data_drive/Macaque_CB/processing/results_from_cell_counts/all_TP_image_idxs_file_lookup.csv')

# missing_idxs_to_fill = [8]
# missing_idxs_to_fill = [3]
# missing_idxs_to_fill = None
all_image_fnames = list(_df['file_name'].values)

# all_image_fnames = all_image_fnames[0:30] #for testing

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

# TODO add deformation_smoothing across the stack by introducing deformation_smoothing_kernel = None here
# and keeping the forward transforms if not none. Smoothing would have to be tackled at the template stage  
def coreg_single_slice_orig(idx, output_dir, subject, img, all_image_names, template, 
                       target_slice_offset_list=[-1, -2, -3], zfill_num=4, 
                       input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl',
                       run_syn=True, run_rigid=True, previous_target_tag=None, 
                       scaling_factor=64, image_weights=None, retain_reg_mappings=False):
    """
    Register a single slice in a stack to its neighboring slices based on specified offsets.

    Parameters:
        idx (int): Index of the slice being registered.
        output_dir (str): Directory to save the output files.
        subject (str): Identifier for the dataset or subject being processed.
        img (str): Path to the current slice image file.
        all_image_names (list): List of all slice image filenames in the stack.
        template (list or str): Template image(s) for registration. If a list, each slice uses its corresponding template.
        target_slice_offset_list (list): List of slice offsets (e.g., `[-1, -2, -3]`) to define neighboring slices for registration.
        zfill_num (int): Zero-padding length for slice indices in filenames.
        input_source_file_tag (str): Suffix for identifying the input source files (e.g., `coreg0nl`).
        reg_level_tag (str): Tag for the output registration level (e.g., `coreg1nl`).
        run_syn (bool): Whether to run non-linear (SyN) registration. (does not include affine)
        run_rigid (bool): Whether to run rigid registration.
        previous_target_tag (str): Suffix of the previous iteration's output to use as input for this step. Defaults to the initial source.
        scaling_factor (int): Scaling factor for the image resolution during registration.
        image_weights (list): Weights assigned to images during registration to emphasize certain slices.
        retain_reg_mappings (bool): If True, retain all of the registration output mappings for later use.
    """

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
    if type(template) is list:
        targets = [template[idx]]
    else:
        targets = [template]
    
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
            
    
    logging.info(f'\n\tslice_idx: {idx}\n\t\tsources: {sources[0].split("/")[-1]}\n\t\ttargets: {[t.split("/")[-1] for t in targets]}\n\t\tweights: {image_weights_ordered}') #source is always the same 

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
    if not retain_reg_mappings:
        def_files = glob.glob(f'{output}_ants-def*')
        for f in def_files:
            if 'def0' not in f:
                os.remove(f)
                time.sleep(0.5)
    logging.warning(f"\t\tRegistration completed for slice {idx}.")

def run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=3, 
                                  target_slice_offset_list=[-1,-2,-3], zfill_num=4, input_source_file_tag='coreg0nl', 
                                  reg_level_tag='coreg1nl', run_syn=True, run_rigid=True, previous_target_tag=None, 
                                  scaling_factor=64, image_weights=None, retain_reg_mappings=False):
    """
    Perform parallel registration for a stack of slices by iteratively aligning each slice with its neighbors.

    The reverse direction uses the same function, but target_slice_offset list is negative to ensure proper lookup.
     - the actual order of in which the registrations are submitted is the same 0 -> n:
    
    Parameters:
        output_dir (str): Directory to save output files.
        subject (str): Identifier for the dataset or subject being processed.
        all_image_fnames (list): List of all slice image filenames in the stack.
        template (list or str): Template image(s) for registration. Can be a single template or one per slice.
        max_workers (int): Maximum number of workers for parallel execution.
        target_slice_offset_list (list): List of slice offsets to define neighboring slices for registration.
        zfill_num (int): Zero-padding length for slice indices in filenames.
        input_source_file_tag (str): Suffix for identifying the input source files (e.g., `coreg0nl`).
        reg_level_tag (str): Tag for the output registration level (e.g., `coreg1nl`).
        run_syn (bool): Whether to run non-linear (SyN) registration.
        run_rigid (bool): Whether to run rigid registration.
        previous_target_tag (str): Suffix of the previous iteration's output to use as input for this step.
        scaling_factor (int): Scaling factor for the image resolution during registration.
        image_weights (list): Weights assigned to images during registration to emphasize certain slices.
        retain_reg_mappings (bool): If True, retain all of the registration output mappings for later use.
    """

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, img in enumerate(all_image_fnames):
            futures.append(
                executor.submit(coreg_single_slice_orig, idx, output_dir, subject, img, all_image_names, template, 
                                target_slice_offset_list=target_slice_offset_list, zfill_num=zfill_num, 
                                input_source_file_tag=input_source_file_tag, reg_level_tag=reg_level_tag,
                                run_syn=run_syn, run_rigid=run_rigid, previous_target_tag=previous_target_tag,
                                scaling_factor=scaling_factor, image_weights=image_weights,
                                retain_reg_mappings=retain_reg_mappings)
            )
        for future in as_completed(futures):
            try:
                future.result()
                # logging.warning("Registration completed for one slice.")
            except Exception as e:
                logging.error(f"Registration failed with error: {e}")

def run_cascading_coregistrations(output_dir, subject, all_image_fnames, anchor_slice_idx = None,
                                  missing_idxs_to_fill = None, zfill_num=4, input_source_file_tag='coreg0nl', 
                                  reg_level_tag='coreg1nl', previous_target_tag=None):
    """
    GENERATED BY CHATGPT TODO:edit this
    Perform a cascading registration of a stack of histology slices to build a consistent 3D volume by sequentially registering 
    each slice to adjacent slices. This process involves a combination of rigid and nonlinear (SyN) transformations to iteratively 
    align slices from a central slice outward.

    Parameters:
    -----------
    output_dir : str
        Directory to save registered output files.

    subject : str
        Subject identifier to include in the output filenames.

    all_image_fnames : list of str
        List of file paths to the original stack of 2D image slices to be registered.

    anchor_slice_idx : int, optional
        Index of the central slice to initiate registration. If None, defaults to the middle slice of the stack.

    missing_idxs_to_fill : list of int, optional
        List of slice indices that are missing from the stack, to avoid using this as the anchor slice.

    zfill_num : int, default=4
        Number of zeroes for zero-padding slice indices in the output filenames (e.g., 0001, 0002).

    input_source_file_tag : str, default='coreg0nl'
        Tag representing the initial registration step, used to identify input files.

    reg_level_tag : str, default='coreg1nl'
        Tag representing the current registration level, used to label output files.

    previous_target_tag : str, optional
        Optional tag to specify a previously registered target for the initial alignment. If None, defaults to `input_source_file_tag`.

    Workflow:
    ---------
    1. Identify the central slice in the stack to start registration.
    2. Initialize output filenames based on `output_dir`, `subject`, and `reg_level_tag`.
    3. Save the central slice to the output file without changes to serve as the initial target.
    4. Define indices to register slices in both directions from the central slice:
       - `rw_idxs` for rightward (increasing index) registration.
       - `lw_idxs` for leftward (decreasing index) registration.
    5. Use a cascading registration approach, sequentially aligning each slice with its neighbor:
       - For each slice, perform a rigid alignment followed by nonlinear (SyN) transformation to refine alignment.
       - Write the registered slice to `all_image_fnames_new` so it can serve as the target for the next slice.
    
    Returns:
    --------
    None
        The function saves registered slices as new `.nii.gz` files in `output_dir`.

    Example:
    --------
    run_cascading_coregistrations(output_dir='/path/to/output/', subject='subject01', 
                                  all_image_fnames=['/path/to/slice1.nii.gz', '/path/to/slice2.nii.gz', ...])

    Notes:
    ------
    This method is particularly useful for stacks with alignment inconsistencies, where anchoring to a central slice 
    and cascading outwards can help mitigate blocky artifacts. Adjust the transformation types and parameters within 
    the `ants.registration` calls as necessary for optimal alignment.

    """
    # params for regularization of nonlinear deformations w/ 'SyNOnly'
        # from nighres, the last two numbers of the syn_param correspond to the flow and total sigmas (fluid and elastic deformations, respectively)
        # if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        # elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        # elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
    syn_flow_sigma = 3 #3 is the default w/ this ants.registration call
    syn_total_sigma = 1 #0 is the default w/ this ants.registration call

    import ants

    if previous_target_tag is not None:
        previous_tail = f'_{previous_target_tag}_ants-def0.nii.gz' #if we want to use the previous iteration rather than building from scratch every time (useful for windowing)
    else:
        previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'

    #identify a central slice to start our registration from, rather than anchoring @ the end
    #but we make sure that it is not a missing slice
    
    if anchor_slice_idx is None:
        anchor_slice_idx = int(numpy.floor(len(all_image_fnames)/2))
    if missing_idxs_to_fill is not None:
        while anchor_slice_idx in missing_idxs_to_fill:
            anchor_slice_idx -= 1
            if anchor_slice_idx < 0:
                raise ValueError("No valid start slice index found in the stack.")
    
    # list of what our outputs will be 
    all_image_fnames_new = []
    for idx in numpy.arange(len(all_image_fnames)):
        img_basename = os.path.basename(all_image_fnames[idx]).split('.')[0]
        all_image_fnames_new.append(f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_basename}_{reg_level_tag}_ants-def0.nii.gz")

    #list of what our .nii inputs should be
    all_image_fnames_nii = []
    for idx in numpy.arange(len(all_image_fnames)):
        img_basename = os.path.basename(all_image_fnames[idx]).split('.')[0]
        all_image_fnames_nii.append(f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_basename}{previous_tail}")

    #load and then save the central slice with the new tag, no change since this is the space we want to align to
    save_volume(all_image_fnames_new[anchor_slice_idx],load_volume(all_image_fnames_nii[anchor_slice_idx]))

    #define leftware and rightward indices, then split to source and targets so that we register 4<-5, 5<-6, ... and 3->4, 2->3, ... 
    rw_idxs = numpy.arange(anchor_slice_idx,len(all_image_fnames_nii))
    lw_idxs = numpy.arange(anchor_slice_idx,-1,-1)

    # this is setup to register adjacent slices to that central slice, then cascade the registrations to the left and right
    rw_src_idxs = rw_idxs[1:]
    rw_trg_idxs = rw_idxs[:-1]
    lw_src_idxs = lw_idxs[1:]
    lw_trg_idxs = lw_idxs[:-1]

    src_idxs = numpy.concatenate((rw_src_idxs,lw_src_idxs))
    trg_idxs = numpy.concatenate((rw_trg_idxs,lw_trg_idxs))


    #run the registrations
    for idx, _  in enumerate(src_idxs):
        # img_basename = os.path.basename(all_image_fnames[idx]).split('.')[0]
        target_idx = trg_idxs[idx]
        moving_idx = src_idxs[idx]

        # in each case, only one source and one target, but we use the same code as above
        source = all_image_fnames_nii[moving_idx]
        target = all_image_fnames_new[target_idx] #targets always come from the new list, since this is where the registrered sources will be (and we pre-filled the start_slice_idx image)
        output = all_image_fnames_new[moving_idx]

        source_img = ants.image_read(source)
        target_img = ants.image_read(target)
        logging.info(f'\n\tslice_idx: {src_idxs[idx]}\n\t\tsources: {source.split("/")[-1]}\n\t\ttarget: {target.split("/")[-1]}\n\t\toutput: {output.split("/")[-1]}') #source is always the same 

        pre_to_post_rigid = ants.registration(fixed=target_img, moving=source_img, type_of_transform='Rigid') #run rigid
        pre_aligned = ants.apply_transforms(fixed=target_img, moving=source_img, transformlist=pre_to_post_rigid['fwdtransforms']) #apply rigid
        pre_to_post_nonlin = ants.registration(fixed=target_img, moving=pre_aligned, 
                                               type_of_transform='SyNOnly') #),
                                            #    flow_sigma=syn_flow_sigma,
                                            #    total_sigma=syn_total_sigma) # run nonlin only
        warpedmovout = pre_to_post_nonlin['warpedmovout']

        ants.image_write(warpedmovout, output)
        logging.warning(f"\t\tCascade registration completed for slice {src_idxs[idx]}.")

def compute_intermediate_non_linear_slice(pre_img, post_img, current_img=None, additional_coreg_mean = True, 
                                          idx=None):
    """
    GENERATED BY CHATGPT TODO:edit this
    Compute an intermediate slice by averaging rigid and non-linear transformations between adjacent slices in a histology stack. 
    This approach ensures smooth transitions and alignment between slices, enabling consistent 3D volume reconstruction. 
    Optional additional coregistration refines the intermediate slice by registering pre/post slices to the computed intermediate 
    slice and averaging them again.

    Workflow Description:
    1. **Rigid Registration**:
    - Perform rigid registration from the pre-slice to the post-slice and vice versa.
    - Align pre- and post-slices using the computed rigid transformations.
    
    2. **Non-Linear Registration**:
    - Perform SyN-only non-linear registration on the rigidly aligned pre- and post-slices.
    - Extract the deformation fields and average them to compute the intermediate transformation.

    3. **Apply Averaged Transformation**:
    - Use the averaged non-linear deformation field to warp the rigidly aligned pre-slice to generate the intermediate slice.

    4. **Optional Additional Coregistration**:
    - Register both pre- and post-slices to the intermediate slice.
    - Refine the intermediate slice by averaging these registered outputs.

    5. **Optional Current Slice Registration**:
    - If a current slice is provided, align it to the refined intermediate slice using rigid (and optionally non-linear) registration.

    Parameters:
    -----------
    pre_img : str
        Filename of the pre-slice image.

    post_img : str
        Filename of the post-slice image.

    current_img : str, optional
        Filename of the current slice image to be registered to the intermediate slice.

    additional_coreg_mean : bool, default=True
        If True, performs additional coregistration of the intermediate slice with the pre- and post-slices for refinement.

    idx : int, optional
        Slice index for tracking during parallelized runs.

    Returns:
    --------
    numpy.ndarray or tuple (int, numpy.ndarray)
        - If `idx` is None: Returns the intermediate slice as a NumPy array.
        - If `idx` is provided: Returns a tuple of the slice index and the intermediate slice as a NumPy array.

    Notes:
    ------
    - Input images must be of the same dimensions and spatial resolution.
    - Rigid registration initializes alignment, while non-linear (SyN) registration refines spatial consistency.
    - The optional `current_img` alignment ensures smoother transitions when the current slice is registered to the intermediate slice.
    - Temporary files are used for saving intermediate results during transformation field application.
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
    # we use full syn b/c we actually want the intermediate image to be the size in between the two (so include affine here)
    pre_to_post_nonlin = ants.registration(fixed=post_aligned, moving=pre_aligned, type_of_transform='SyNOnly') #here we do full SyN, effectively rigid + affine + nonlin 
    post_to_pre_nonlin = ants.registration(fixed=pre_aligned, moving=post_aligned, type_of_transform='SyNOnly')

    # Step 4: Load the non-linear deformation fields as images
    # https://antspy.readthedocs.io/en/latest/registration.html #for reference
    pre_to_post_field = ants.image_read(pre_to_post_nonlin['fwdtransforms'][0]) # 2nd on the stack is the Affine
    post_to_pre_field = ants.image_read(post_to_pre_nonlin['invtransforms'][1]) # grab the inverse deformation field, which is the 2nd on the stack for inverted


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

    intermediate_img_np = intermediate_img.numpy()

    # if you want to, we now coregister the images to the new target and then take their mean
    if additional_coreg_mean: 
        # Step 1: Perform rigid registration from pre to post slice and post to pre slice
        pre_to_post_rigid = ants.registration(fixed=intermediate_img, moving=pre_ants, type_of_transform='Rigid')
        post_to_pre_rigid = ants.registration(fixed=intermediate_img, moving=post_ants, type_of_transform='Rigid')

        # Step 2: Apply the rigid transformation to each image for initial alignment
        pre_aligned = ants.apply_transforms(fixed=intermediate_img, moving=pre_ants, transformlist=pre_to_post_rigid['fwdtransforms'])
        post_aligned = ants.apply_transforms(fixed=intermediate_img, moving=post_ants, transformlist=post_to_pre_rigid['fwdtransforms'])


        # # Step 3: Perform non-linear registration on the rigidly aligned images
        # we use full syn b/c we actually want the intermediate image to be the size in between the two (so include affine here)
        pre_to_post_nonlin = ants.registration(fixed=intermediate_img, moving=pre_aligned, type_of_transform='SyNOnly')
        post_to_pre_nonlin = ants.registration(fixed=intermediate_img, moving=post_aligned, type_of_transform='SyNOnly')

        # Step 4: Load the deformed images
        pre_to_post_img = pre_to_post_nonlin['warpedmovout']
        post_to_pre_img = post_to_pre_nonlin['warpedmovout']

        # Step 5: Compute the average of the deformed images
        intermediate_img_np = (pre_to_post_img.numpy() + post_to_pre_img.numpy()) / 2

        if current_img is not None: #if we have a current image to push into this space, we should do this here
            current_img = ants.image_read(current_img)
            # Step 6: Register the current slice to the interpolated slice
                            
            # Convert the data array to an ANTs image
            new_image = ants.from_numpy(intermediate_img_np)
            # Set the spatial information (origin, spacing, direction) from the reference image
            new_image.set_origin(pre_ants.origin)
            new_image.set_spacing(pre_ants.spacing)
            new_image.set_direction(pre_ants.direction)
            # ants.image_write(new_image, intermediate_img_fname)
            
            
            #rigid
            current_to_template_rigid = ants.registration(fixed=new_image,moving=current_img,type_of_transform='Rigid') 
            current_aligned_rigid = ants.apply_transforms(fixed=new_image, moving=current_img, transformlist=current_to_template_rigid['fwdtransforms'])
            
            ## TODO: uncertain if this is necessary, as incorporating the nonlin step here may hurt more than help, since we are deforming to the intermediate img
            #nonlin
            current_to_template_nonlin = ants.registration(fixed=new_image,moving=current_aligned_rigid,type_of_transform='SyNOnly')
            new_intermediate_img = current_to_template_nonlin['warpedmovout']
                
            intermediate_img_np = new_intermediate_img.numpy()

    if idx is not None: #if we passed an index value, this is to keep track of parallel so we pass it back
        return idx, intermediate_img_np
    else: 
        return intermediate_img_np



def generate_missing_slices(missing_fnames_pre,missing_fnames_post,current_fnames=None,method='intermediate_nonlin_mean',nonlin_interp_max_workers=1):
    """
    Parallelized generation of missing slices in a histology stack by interpolating between adjacent slices using specified methods. 
    This function supports simple averaging or advanced interpolation using non-linear transformations, making it 
    versatile for handling gaps in data.

    Workflow Description:
    1. **Input Parsing**:
    - Accepts lists of filenames for the pre-slice, post-slice, and optionally the current slice (for matching indices).

    2. **Interpolation Methods**:
    - **Mean**:
        - Directly averages the corresponding pre- and post-slices to generate the missing slice.
    - **Intermediate Non-Linear Mean**:
        - Applies non-linear transformations to the neighboring pre- and post-slices.
        - Averages the deformation fields and computes the interpolated slice using the computed intermediate transformation.
        - Supports parallel processing for efficient computation.

    3. **Parallelized Execution**:
    - For the intermediate non-linear mean method, the computation for each slice is dispatched to a process pool for parallel execution.
    - Results are gathered, sorted by slice index, and stacked into a consistent 3D volume.

    Parameters:
    -----------
    missing_fnames_pre : list of str
        List of filenames for the pre-slices corresponding to missing slices.

    missing_fnames_post : list of str
        List of filenames for the post-slices corresponding to missing slices.

    current_fnames : list of str, optional
        List of filenames for the current slices (if available), aligned to the missing slice indices.

    method : str, default='intermediate_nonlin_mean'
        Method for generating missing slices:
        - 'mean': Simple averaging of neighboring slices.
        - 'intermediate_nonlin_mean': Uses non-linear transformations and averaging of deformation fields.

    nonlin_interp_max_workers : int, default=1
        Maximum number of workers to use for parallel non-linear interpolation.

    Returns:
    --------
    numpy.ndarray
        A 3D stack of interpolated missing slices, where each slice is generated based on the specified method.

    Notes:
    ------
    - Ensure that the input lists (`missing_fnames_pre`, `missing_fnames_post`, and optionally `current_fnames`) 
    are aligned in order and size.
    - The 'intermediate_nonlin_mean' method leverages `compute_intermediate_non_linear_slice` for advanced interpolation.
    - Parallel execution significantly reduces computation time for large stacks.
    - The `mean` method is simpler but may not handle nonlinear spatial variations between slices effectively.
    - Temporary variables (e.g., `the_slices`) are reordered by slice index to ensure correct stacking of the output volume.
    """

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
            
    elif method == 'intermediate_nonlin_mean':
        futures = []
        the_idxs = []
        the_slices = []
        with ProcessPoolExecutor(max_workers=nonlin_interp_max_workers) as executor:
            for idx, _ in enumerate(missing_fnames_pre):
                img_fname_pre = missing_fnames_pre[idx]
                img_fname_post = missing_fnames_post[idx]
                if current_fnames is not None:
                    img_fname_current = current_fnames[idx]
                else:
                    img_fname_current=None

                futures.append(executor.submit(
                    compute_intermediate_non_linear_slice,
                    pre_img=img_fname_pre,
                    post_img=img_fname_post,
                    current_img=img_fname_current,
                    idx=idx
                ))
            for future in as_completed(futures):
                try:
                    the_idx, the_slice = future.result()
                    the_idxs.append(the_idx)
                    the_slices.append(the_slice)
                    logging.warning(f'\t\tParallel slice generation completed for slice: {the_idx}')
                except Exception as e:
                    logging.warning('Parallel slice generation failed: {e}')
        idxs_order = numpy.argsort(the_idxs)
        sorted_slices = [the_slices[i] for i in idxs_order]
        missing_slices_interpolated= numpy.stack(sorted_slices, axis=-1) #reorder based on the indices that were passed
    else:
        missing_slices_interpolated = None
    return missing_slices_interpolated

def generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=4,reg_level_tag='coreg12nl',
                                per_slice_template=False,missing_idxs_to_fill=None, slice_template_type='median',nonlin_interp_max_workers=1):
    """
    TODO: update with better version of ChatGPT! 
    Generate a stack of registered slices and create either a single median template or template image for each slice.
    The output image stack is the collection of all registered slices, with missing slices filled in through interpolation between neighboring slices. The corresponding templates are also generated, including a median template and a non-linear version of the mean of slice positions [-1,0,1].

    Args:
        output_dir (str): Directory to store the output files.
        subject (str): The subject identifier (e.g., subject name or ID).
        all_image_fnames (list): List of filenames corresponding to the registered image slices.
        zfill_num (int, optional): Number of digits to use for zero-padding slice indices in filenames (default is 4).
        reg_level_tag (str, optional): Tag indicating the registration level (default is 'coreg12nl').
        per_slice_template (bool, optional): If True, generate individual templates for each slice (default is False).
        missing_idxs_to_fill (list, optional): List of indices corresponding to missing slices to be filled by interpolation (default is None).
        slice_template_type (str or list, optional): Specifies the method for generating the slice templates. Can be 'mean', 'median', 'nonlin', or a list of these methods (default is 'median').
        nonlin_interp_max_workers (int, optional): Number of workers to use for parallelized non-linear interpolation (default is 1).

    Returns:
        str or list: The filename(s) of the generated template(s). If `per_slice_template` is True, a list of template filenames (both median and non-linear templates) is returned. Otherwise, the filename of the median template is returned.

    Notes:
        - If `per_slice_template` is True, a separate template is generated for each slice in the stack.
        - Missing slices, specified by `missing_idxs_to_fill`, are filled using the 'intermediate_nonlin_mean' interpolation method by default.
        - The slice templates are generated using the specified `slice_template_type` method. If 'nonlin' is chosen, non-linear interpolation is applied to the slices.

    Workflow:
        1. **Loading the Image Slices**:
            - The image slices are loaded from the list of filenames `all_image_fnames`.
            - These slices are assumed to be pre-registered.
        
        2. **Handling Missing Slices**:
            - If `missing_idxs_to_fill` is provided, it indicates the indices of missing slices to be interpolated.
            - For each missing slice, corresponding pre-slice and post-slice filenames are identified from `all_image_fnames` based on the indices.
            - Missing slices are filled by interpolation using neighboring slices. The default interpolation method is 'intermediate_nonlin_mean'.

        3. **Stack Generation**:
            - The registered image slices are stacked into a 3D array, and missing slices are filled with interpolated values.
            - The result is saved as the image stack file (`_stack.nii.gz`).

        4. **Template Generation**:
            - A single template image is generated as the median of all registered slices. This is saved as `_template.nii.gz`.
            - Optionally, slice-wise templates can be generated using the `slice_template_type` argument. Possible options include:
                - `'mean'`: The mean of the current and neighboring slices.
                - `'median'`: The median of the current and neighboring slices.
                - `'nonlin'`: A non-linear version of the template using the 'intermediate_nonlin_mean' method.
            
        5. **Saving Slice Templates**:
            - If `per_slice_template` is set to True, templates for each slice are saved individually, both for median and non-linear methods, as applicable.

        6. **Return Value**:
            - The function returns the filename(s) of the generated templates. If individual slice templates are generated, a list of template filenames is returned; otherwise, a single template filename is returned.
    """
    
    #we can also output a per_slice_template based on the median of the current and neighbouring slices
    stack = []
    stack_tail = f'_{reg_level_tag}_stack.nii.gz'
    img_stack = output_dir+subject+stack_tail
    img_stack_nonlin = img_stack.replace('.nii.gz','_nonlin.nii.gz')
    template_tail = f'_{reg_level_tag}_template.nii.gz'
    template_nonlin_tail = f'_{reg_level_tag}_template_nonlin.nii.gz'
    template = output_dir+subject+template_tail

    template_list = []
    template_nonlin_list = []

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
                logging.warning(f"Processing missing slice idx: {img_idx+1}") #add one, b/c we are looping over the pre slice 
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

            for idx, img_idx in enumerate(missing_idxs_to_fill):
                if idx == 0:
                    missing_fnames_current = []
                img_name = all_image_fnames[img_idx]
                img_name = os.path.basename(img_name).split('.')[0]
                reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
                missing_fnames_current.append(reg)

            missing_slices_interpolated = generate_missing_slices(missing_fnames_pre,
                                                                  missing_fnames_post,
                                                                  method='intermediate_nonlin_mean',
                                                                  nonlin_interp_max_workers=nonlin_interp_max_workers)

            
            #now we can fill the slices with the interpolated value
            for idx,missing_idx in enumerate(missing_idxs_to_fill):
                #we overwrite the missing slices with the interpolated values and insert it into the img stack
                missing_fname = missing_fnames_current[idx]
                interp_slice = missing_slices_interpolated[...,idx]
                header = nibabel.Nifti1Header()
                header.set_data_shape(interp_slice.shape)
                nifti = nibabel.Nifti1Image(interp_slice,affine=None,header=header)
                nifti.update_header()
                save_volume(missing_fname,nifti)
                # print(idx)
                # print(missing_idx)
                # print(numpy.shape(missing_slices_interpolated))
                # print(numpy.shape(img))
                img[...,missing_idx] = interp_slice

        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(img_stack,nifti)

        ## if requested, we output templates for each slice based on the median of the surrounding slices (-1,1)
        # for slices within missing_idxs_to_fill we already interpolated them, so we leave these as is
        if per_slice_template:
            num_slices = img.shape[-1]
            if not (type(slice_template_type) == list):
                slice_template_type = [slice_template_type]
            if 'median' in slice_template_type:
                logging.warning('Generating median slice templates')
                for idx,img_name in enumerate(all_image_fnames):
                    img_name = os.path.basename(img_name).split('.')[0]
                    slice_template_fname = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_tail
                    if idx == 0: #if at the front, take the first two only
                        slice_template = numpy.median(img[...,0:2],axis=-1)
                    elif idx == num_slices-1: #if at the end, take the last two only
                        slice_template = numpy.median(img[...,-2:],axis=-1)
                    elif missing_idxs_to_fill is not None and idx in missing_idxs_to_fill:
                        logging.warning(f'========> NOT COMPUTING THE MEDIAN FOR MISSING SLICE at idx {idx}')
                        slice_template = img[...,idx]
                    else: #take one on each side and the current slice
                        start = idx-1
                        stop = idx+2
                        slice_template = numpy.median(img[...,start:stop],axis=-1)
            
                    header.set_data_shape(slice_template.shape)
                    nifti = nibabel.Nifti1Image(slice_template,affine=None,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname,nifti)
                    template_list.append(slice_template_fname)            
      
            elif 'mean' in slice_template_type:
                logging.warning('Generating mean slice templates')
                for idx,img_name in enumerate(all_image_fnames):
                    img_name = os.path.basename(img_name).split('.')[0]
                    slice_template_fname = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_tail
                    if idx == 0: #if at the front, take the first two only
                        slice_template = numpy.mean(img[...,0:2],axis=-1)
                    elif idx == num_slices-1: #if at the end, take the last two only
                        slice_template = numpy.mean(img[...,-2:],axis=-1)
                    elif missing_idxs_to_fill is not None and idx in missing_idxs_to_fill:
                        slice_template = img[...,idx]
                    else: #take one on each side and the current slice
                        start = idx-1
                        stop = idx+2
                        slice_template = numpy.mean(img[...,start:stop],axis=-1)
            
                    header.set_data_shape(slice_template.shape)
                    nifti = nibabel.Nifti1Image(slice_template,affine=None,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname,nifti)
                    template_list.append(slice_template_fname)            

            if 'nonlin' in slice_template_type:
                logging.warning('Generating non-linear slice templates')
                
                # first write the slices "as is" from our stack, using our template_nonlin_tail
                # these slice templates will be overwritten with the interpolated versions
                for idx,img_name in enumerate(all_image_fnames):
                    img_name = os.path.basename(img_name).split('.')[0]
                    slice_template_fname = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_nonlin_tail
                    slice_template = img[...,idx]

                    header.set_data_shape(slice_template.shape)
                    nifti = nibabel.Nifti1Image(slice_template,affine=None,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname,nifti)
                    template_nonlin_list.append(slice_template_fname)
                
                # then use these images to generate the interpolations, skipping the first and last images (as anchors) 
                missing_fnames_pre_1 = template_nonlin_list[:-2] #starting from 0, skip the last two (b/c last has no pair), treated as pre
                missing_fnames_post_1 = template_nonlin_list[2:] #starting from 1 (skip the first one), treated as pre
                interp_template_slices = generate_missing_slices(missing_fnames_pre_1,missing_fnames_post_1,
                                                                 current_fnames=template_nonlin_list[1:-1],
                                                                 method='intermediate_nonlin_mean',
                                                                 nonlin_interp_max_workers=nonlin_interp_max_workers)

                #fill the image stack with the interpolated slices
                # save with a differnt fname so that we can see what this looks like
                img[...,1:-1] = interp_template_slices
                nifti = nibabel.Nifti1Image(img,affine=None,header=header)
                save_volume(img_stack_nonlin,nifti)

                #save them as their own individual templates, saving over the original ones that were not yet interpolated
                for idx,slice_template_fname_nonlin in enumerate(template_nonlin_list):
                    slice_template = img[...,idx]
                    header.set_data_shape(slice_template.shape)
                    nifti = nibabel.Nifti1Image(slice_template,affine=None,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname_nonlin,nifti)
            
        #now save the single template (as a median only)
        img = numpy.median(img,axis=2)
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(template,nifti)
        print('Stacking: done - {}'.format(template))
    
    #return the template filename(s)
    if per_slice_template:
        if len(slice_template_type) == 1:
            return template_list
        else:
            return template_list, template_nonlin_list
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


def compute_MI_for_slice(idx, img_name, output_dir, subject, template_tail, out_tail, tag1_tail, 
                         tag2_tail, zfill_num, per_slice_template, overwrite):
    from sklearn.metrics import mutual_info_score
    
    img_name = os.path.basename(img_name).split('.')[0]
    if not per_slice_template:
        template_path = output_dir + subject + template_tail
    else:
        template_path = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + template_tail
    
    output_path = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + out_tail + '_ants-def0.nii.gz'
    if os.path.isfile(output_path) and not overwrite:
        return None

    slice1_path = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + tag1_tail + '_ants-def0.nii.gz'
    slice2_path = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + tag2_tail + '_ants-def0.nii.gz'
    
    curr1 = nighres.io.load_volume(slice1_path).get_fdata()
    curr2 = nighres.io.load_volume(slice2_path).get_fdata()
    curr = nighres.io.load_volume(template_path).get_fdata()
    
    # Flatten the images
    curr1_flat = curr1.flatten()
    curr2_flat = curr2.flatten()            
    curr_flat = curr.flatten()

    # Define bin edges based on the data range
    min_val = min(curr1_flat.min(), curr2_flat.min(), curr_flat.min())
    max_val = max(curr1_flat.max(), curr2_flat.max(), curr_flat.max())
    bins = numpy.linspace(min_val, max_val, 100) #100 bins

    # Digitize each array to convert to discrete bins
    curr1_binned = numpy.digitize(curr1_flat, bins)
    curr2_binned = numpy.digitize(curr2_flat, bins)
    curr_binned = numpy.digitize(curr_flat, bins)

    mi1c = mutual_info_score(curr1_binned, curr_binned)
    mi2c = mutual_info_score(curr2_binned, curr_binned)

    # Copy the best result
    mapping = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + out_tail + '_ants-map.nii.gz'
    inverse = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + out_tail + '_ants-invmap.nii.gz'
    
    if mi1c > mi2c:
        mapping1 = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + tag1_tail + '_ants-map.nii.gz'
        inverse1 = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + tag1_tail + '_ants-invmap.nii.gz'
        shutil.copyfile(mapping1, mapping)
        shutil.copyfile(inverse1, inverse)
        shutil.copyfile(slice1_path, output_path)
    else:
        mapping2 = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + tag2_tail + '_ants-map.nii.gz'
        inverse2 = output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + tag2_tail + '_ants-invmap.nii.gz'
        shutil.copyfile(mapping2, mapping)
        shutil.copyfile(inverse2, inverse)
        shutil.copyfile(slice2_path, output_path)
    
    # Clean up old files
    for f in glob.glob(output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + "*_ants-*map.nii.gz"):
        if out_tail not in f:
            os.remove(f)
            time.sleep(0.5)
    os.remove(slice1_path)
    time.sleep(0.5)
    os.remove(slice2_path)
    time.sleep(0.5)
    
    return idx, mapping, img_name, mi1c, mi2c

def select_best_reg_by_MI_parallel(output_dir, subject, all_image_fnames, df_struct=None, template_tag='coreg0nl',
                                   zfill_num=3, reg_level_tag1='coreg1nl', reg_level_tag2='coreg2nl',
                                   reg_output_tag='coreg12nl', per_slice_template=False, overwrite=True, use_nonlin_slice_templates=False,
                                   max_workers=1):
    '''
    Use MI to determine best registration (forwards or backwards) and select going forward
    '''
    
    if use_nonlin_slice_templates:
        template_tail = f'_{template_tag}_template_nonlin.nii.gz'
    else:
        template_tail = f'_{template_tag}_template.nii.gz'
    
    out_tail = f'_{reg_output_tag}'
    tag1_tail = f'_{reg_level_tag1}'
    tag2_tail = f'_{reg_level_tag2}'
    
    results = []
    args = [
        (idx, img_name, output_dir, subject, template_tail, out_tail, tag1_tail, tag2_tail, zfill_num, per_slice_template, overwrite)
        for idx, img_name in enumerate(all_image_fnames)
    ]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(compute_MI_for_slice, *arg): arg[1] for arg in args}
        for future in as_completed(future_to_img):
            result = future.result()
            if result:
                results.append(result)
    
    
    #convert from tuple to list of lists, then sort by index
    sorted_results = [list(r) for r in results]
    sorted_results.sort(key=lambda x: x[0])
    if df_struct is not None:
        df_struct['idx'] = [r[0] for r in sorted_results]
        df_struct['selected_transform'] = [r[1] for r in sorted_results]
        df_struct['img_name' + out_tail] = [r[2] for r in sorted_results]
        df_struct['MI1c' + out_tail] = [r[3] for r in sorted_results]
        df_struct['MI2c' + out_tail] = [r[4] for r in sorted_results]
        return df_struct
    else:
        return None


def select_best_reg_by_MI(output_dir,subject,all_image_fnames,df_struct=None, template_tag='coreg0nl',
                          zfill_num=zfill_num,reg_level_tag1='coreg1nl', reg_level_tag2='coreg2nl',reg_output_tag='coreg12nl',per_slice_template=False,
                          overwrite=True):
    '''
    Use MI to determine best registration (forwards or backwards) and select going forward
    reg_output_tag identifies the best registration outputs
    '''
    from sklearn.metrics import mutual_info_score
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
            
            # p1,v1 = numpy.histogram(curr1.flatten(), bins=100, density=True)
            # p2,v2 = numpy.histogram(curr2.flatten(), bins=100, density=True)
            # pc,vc = numpy.histogram(curr.flatten(), bins=100, density=True)

            # # normalize histograms to 1
            # p1 = p1/numpy.sum(p1)
            # p2 = p2/numpy.sum(p2)
            # pc = pc/numpy.sum(pc)
            
            # p1c,v1,vc = numpy.histogram2d(curr1.flatten(), curr.flatten(), bins=100, density=True)
            # p2c,v2,vc = numpy.histogram2d(curr2.flatten(), curr.flatten(), bins=100, density=True)
        
            # # normalize joint histograms to 1
            # p1c = p1c / numpy.sum(p1c)
            # p2c = p2c / numpy.sum(p2c)
            
            # p1pc = numpy.outer(p1,pc)
            # p2pc = numpy.outer(p2,pc)
                
            # mi1c = numpy.sum(p1c*numpy.log(p1c/(p1pc),where=(p1c*p1pc>0)))
            # mi2c = numpy.sum(p2c*numpy.log(p2c/(p2pc),where=(p2c*p2pc>0)))
            
            # Flatten the images
            curr1_flat = curr1.flatten()
            curr2_flat = curr2.flatten()            
            curr_flat = curr.flatten()

            # Define bin edges based on the data range
            min_val = min(curr1_flat.min(), curr2_flat.min(), curr_flat.min())
            max_val = max(curr1_flat.max(), curr2_flat.max(), curr_flat.max())
            bins = numpy.linspace(min_val, max_val, 100) #100 bins
    
            # Digitize each array to convert to discrete bins
            curr1_binned = numpy.digitize(curr1_flat, bins)
            curr2_binned = numpy.digitize(curr2_flat, bins)
            curr_binned = numpy.digitize(curr_flat, bins)

            mi1c = mutual_info_score(curr1_binned, curr_binned)
            mi2c = mutual_info_score(curr2_binned, curr_binned)

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
        df_struct['img_name'+out_tail] = img_name_l
        df_struct['MI1c'+out_tail] = mi1c_l
        df_struct['MI2c'+out_tail] = mi2c_l
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
    
    if rescale <=1:
        return image
    else:
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

    if rescale <=1 and prop_pad == 0:
        return image
    else:
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

    if rescale <=1:
        return image
    else:
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

def generate_slice_mask(img_data, threshold_pct = 5):
    '''
    Generate a mask for the slice based on the threshold percentage
    '''
    vec = img_data.flatten()
    cut = numpy.percentile(vec[vec>=1],threshold_pct)
    mask = numpy.zeros(img_data.shape,bool)
    mask[img_data>=cut] = True
    return mask.astype(int)

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
shutil.copyfile(__file__,os.path.join(output_dir,script_name))
logger.info(f'Original script file copied to output directory.')

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

            # exceptions that need fixing, since rigid reg does not seem to address big flips
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
                 nighres.registration.embedded_antspy_2d_multi,source_images=sources, 
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
                    file_name=output
            )
        )
                                    

template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag='coreg0nl',
                                       missing_idxs_to_fill=missing_idxs_to_fill)

## loop over cascades to see what this does for us
# iter_tag = ""
# num_cascade_iterations = 5
# anchor_slice_idxs = numpy.linspace(0,len(all_image_fnames)-1,num_cascade_iterations+2).astype(int)
# anchor_slice_idxs = anchor_slice_idxs[1:-1] #remove the first and last, as they will denote 1st and last indices of the stack
# for iter in range(num_cascade_iterations):
#     if iter == 0:
#         input_source_file_tag = 'coreg0nl'

#     else:
#         input_source_file_tag = iter_tag #updates with the previous iteration
#     iter_tag = f'_cascade_{iter}'
#     run_cascading_coregistrations(output_dir, subject, 
#                                 all_image_fnames, anchor_slice_idx = anchor_slice_idxs[iter], 
#                                 missing_idxs_to_fill = missing_idxs_to_fill, 
#                                 zfill_num=zfill_num, input_source_file_tag=input_source_file_tag, 
#                                 reg_level_tag=iter_tag, previous_target_tag=None)

#     template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag=iter_tag,
#                                         per_slice_template=True,
#                                         missing_idxs_to_fill=missing_idxs_to_fill)

input_source_file_tag = 'coreg0nl'
reg_level_tag = "coreg0nl_cascade"
run_cascading_coregistrations(output_dir, subject, 
                              all_image_fnames, anchor_slice_idx = None, 
                              missing_idxs_to_fill = missing_idxs_to_fill, 
                              zfill_num=zfill_num, input_source_file_tag=input_source_file_tag, 
                              reg_level_tag=reg_level_tag, previous_target_tag=None)

template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=zfill_num,reg_level_tag=reg_level_tag,
                                per_slice_template=True,
                                missing_idxs_to_fill=missing_idxs_to_fill)
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

logger.warning('3. Begin STAGE1 registration iterations - Rigid + Syn')

# STEP 1: Rigid + Syn
num_reg_iterations = 5
run_rigid = True
run_syn = True
template_tag = 'coreg0nl' #initial template tag, which we update with each loop
template_tag = 'coreg0nl_cascade'
MI_df_struct = {} #output for MI values, will be saved in a csv file
# TODO: 1. Test between slice registrations as a way to refine stack
# TODO: 2. Add masks to the registration process to improve speed (hopefully) and precision


for iter in range(num_reg_iterations): 
    
    #here we always go back to the original coreg0 images, we are basically just refning our target template(s)
    
    iter_tag = f"_rigsyn_{iter}"
    print(f'\t iteration tag: {iter_tag}')
    logger.warning('****************************************************************************')
    logger.warning(f'\titeration {iter_tag}')
    logger.warning('****************************************************************************')
    
    if (iter == 0): #do not want to use per slice templates
        # first_run_slice_template = False #skip using the per slice template on the first 2 reg steps below (up until the next template is created), same for use_nonlin_slice_templates
        first_run_slice_template = True
        first_run_nonlin_slice_template = use_nonlin_slice_templates
    else:
        first_run_slice_template = per_slice_template
        first_run_nonlin_slice_template = use_nonlin_slice_templates

    missing_idxs_to_fill = None #XXX FOR TESTING!!! TODO: ONLY FILL IN MISSING SLICES prior to this.

    slice_offset_list_forward = [-1,-2,-3]
    slice_offset_list_reverse = [1,2,3]
    image_weights = generate_gaussian_weights([0,1,2,3],gauss_std=3) #symmetric gaussian, so the same on both sides
    # XXX removed image weights
    image_weights = numpy.ones(len(slice_offset_list_forward)+1)
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers, 
                                target_slice_offset_list=slice_offset_list_forward, 
                zfill_num=zfill_num, input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl'+iter_tag,
                image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers, 
                                target_slice_offset_list=slice_offset_list_reverse, 
                        zfill_num=zfill_num, input_source_file_tag='coreg0nl', reg_level_tag='coreg2nl'+iter_tag,
                        image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)

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
                                            nonlin_interp_max_workers=nonlin_interp_max_workers)
    else:
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                            zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                            missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                            nonlin_interp_max_workers=nonlin_interp_max_workers)
    if use_nonlin_slice_templates:
        template = template_nonlin
    # missing_idxs_to_fill = None #if we only want to fill in missing slices on the first iteration, then we just use that image as the template
    
    ## TODO: insert in here the code to register the stack to the MRI template and then update the tag references as necessary
    # if iter > 0: #we do not do this on the first iteration
        # MRI_reg_output = register_stack_to_mri(slice_stack_template, mri_template)


    template_tag = 'coreg12nl'+iter_tag
    

    ## No diff between these two approaches
    # slice_offset_list_forward = [-3,-2,-1,1,2] #weighted back, but also forward
    # slice_offset_list_reverse = [-2,-1,1,2,3] #weighted forward, but also back
    # same as above
    # slice_offset_list_forward = [-3,-2,-1,1] #weighted back, but also forward
    # slice_offset_list_reverse = [-1,1,2,3] #weighted forward, but also back
    # below is worse
    #slice_offset_list_forward = [-1,-2,-3,-4,-5] 
    #slice_offset_list_reverse = [1,2,3,4,5] 

    #not much change, likely worse    
    # slice_offset_list_forward = [-1] 
    # slice_offset_list_reverse = [1] 

    #increasing the gaussian weigting also results in worse (3-> 5)

    # her we include neigbouring slices and increase the sharpness of the gaussian
    slice_offset_list_forward = [-4,-3,-2,-1,1] #weighted back, but also forward
    slice_offset_list_reverse = [-1,1,2,3,4] #weighted forward, but also back
    image_weights_win1 = generate_gaussian_weights([0,] + slice_offset_list_forward, gauss_std=2) #symmetric gaussian, so the same on both sides
    image_weights_win2 = generate_gaussian_weights([0,] + slice_offset_list_reverse, gauss_std=2)
    # XXX removed image weights
    image_weights_win1 = numpy.ones(len(slice_offset_list_forward)+1)
    image_weights_win2 = numpy.ones(len(slice_offset_list_forward)+1)
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers,
                                target_slice_offset_list=slice_offset_list_forward, 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', 
                    previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win1'+iter_tag,
                    image_weights=image_weights_win1,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    
    run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=max_workers,
                                target_slice_offset_list=slice_offset_list_reverse, 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', 
                    previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win2'+iter_tag,
                    image_weights=image_weights_win2,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    logging.warning('\t\tSelecting best registration by MI')                                     

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
                                            nonlin_interp_max_workers=nonlin_interp_max_workers)
    else:
        template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                            zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                            missing_idxs_to_fill=missing_idxs_to_fill, slice_template_type=slice_template_type,
                                            nonlin_interp_max_workers=nonlin_interp_max_workers)
    
    if use_nonlin_slice_templates:
        template = template_nonlin
    template_tag = 'coreg12nl_win12'+iter_tag
    
final_reg_level_tag = 'coreg12nl_win12'+iter_tag
step1_iter_tag = iter_tag

print(f"Output directory: {output_dir}")

## TODO: ADAPT AFTER ABOVE WORKING

# # # # STEP 2: Syn only
# print('4. Begin STAGE2 registration iterations - Syn')
# logger.warning('4. Begin STAGE2 registration iterations - Syn')
# run_rigid = False
# run_syn = True
# num_syn_reg_iterations = 5
# for iter in range(num_syn_reg_iterations):
#     #for the nonlinear step, we base our registrations on the previous ones instead of going back to the original images, starting with the previous step and 
#     # then using the output from each successive step
#     iter_tag = f"{step1_iter_tag}_syn_{iter}"
#     print(f'\t iteration tag: {iter_tag}')
#     logger.warning(f'\titeration {iter_tag}')

#     slice_offset_list_forward = [-1,-2,-3] #weighted back
#     slice_offset_list_reverse = [1,2,3] #weighted forward
#     image_weights = generate_gaussian_weights([0,1,2,3])
    
#     run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='forward', max_workers=max_workers, 
#                                  target_slice_offset_list=slice_offset_list_forward, 
#                 zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg1nl'+iter_tag,
#                 image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
#     run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, direction='reverse', max_workers=max_workers, 
#                                  target_slice_offset_list=slice_offset_list_reverse, 
#                         zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg2nl'+iter_tag,
#                         image_weights=image_weights,run_syn=run_syn,run_rigid=run_rigid,scaling_factor=scaling_factor)
    
#     logging.warning('\t\tSelecting best registration by MI')    
#     select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
#                         zfill_num=zfill_num,reg_level_tag1='coreg1nl'+iter_tag, reg_level_tag2='coreg2nl'+iter_tag,
#                         reg_output_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,df_struct=MI_df_struct)
#     if MI_df_struct is not None:
#         pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
    
#     logging.warning('\t\tGenerating new template')
#     template = generate_stack_and_template(output_dir,subject,all_image_fnames,
#                                         zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
#                                         missing_idxs_to_fill=missing_idxs_to_fill)
#     template_tag = 'coreg12nl'+iter_tag

    # if MI_df_struct is not None:
    #     pd.DataFrame(MI_df_struct).to_csv(output_dir+subject+'_MI_values.csv',index=False)
    #     # MI_df_struct.to_csv(output_dir+subject+'_MI_values.csv',index=False)
