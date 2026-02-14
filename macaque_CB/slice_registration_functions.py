import os
import time
import shutil
import logging
import sys
from datetime import datetime
import nighres
import numpy
np = numpy #for shorthand
import nibabel
import glob
from PIL import Image
import pandas as pd
# from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, laplace
from scipy.stats import trim_mean
from skimage.exposure import match_histograms
import math
from nighres.io import load_volume, save_volume
from concurrent.futures import ProcessPoolExecutor, as_completed

import tempfile

# code by @pilou, using nighres; adapted, modularized, extended, and parallelized registrations by @csteele
## Potential list of todo's
# TODO: additional weight of registrations by MI to downweight slices that are much different (much more processing)
# TODO: potentially incorporate mesh creation to either identify mask (limiting registration)
#       potentially included as a distance map in some way to weight boundary?
# TODO: fix nonlinear slice interpolation option for template creation
# TODO: test for existing files and do not overwrite by default
# TODO: add steps to change regularization along the way
# TODO: potentially scale regularization
# TODO: potentially filter in slice dir
# nb.Nifti1Image(scipy.signal.savgol_filter(image.get_fdata(), window_length=5, polyorder=2, axis=2),image.affine,image.header).to_fil
#     ...: ename(template_f.split('.')[0]+"_savgol_filt.nii.gz")


def compute_histogram_matched_slice(current_slice, pre_img_slice, post_img_slice = None):
    """
    Compute a histogram-matched slice for the current slice from one or both of its neighboring slices.
    All inputs should be numpy arrays in the same space since we mask based on the current slice
      - this approach intrinsically misses some data in the neighbours, but has little effect

    Parameters:
        current_slice (ndarray): Current slice to be matched to its neigbours
        pre_img_slice (ndarray): Image before current slice.
        post_img_slice (ndarray): Image after current slice.
    """
    
    current_matched = np.zeros_like(current_slice)
    m = current_slice>0
    current_slice_vec = current_slice[m]
    pre_vec = pre_img_slice[m]

    if post_img_slice is not None:
        # Match current slice to the average of its neighbors
        post_vec = post_img_slice[m]
        matched = match_histograms(current_slice_vec, (pre_vec + post_vec) / 2)
        current_matched[m] = matched
    else:
        #match to the one that was provided
        matched = match_histograms(current_slice_vec, pre_vec)
        current_matched[m] = matched
    return current_matched

def compute_scaling_multipliers_from_dataset(image_list, mask_zero=True, trim_proportion=0.05):
    """
    Computes global scaling multipliers from a dataset to normalize sharpening parameters.

    Parameters:
        image_list (list of nibabel images): List of images to compute statistics from
        mask_zero (bool): Whether to exclude zero values from the computation.
        trim_proportion (float): Proportion of values to trim from the high and low ends of the distribution.

    Returns:
        sigma_multiplier (float): Multiplier to convert detail energy into sigma.
        strength_multiplier (float): Multiplier to convert contrast into sharpening strength.
        stats (dict): Dictionary of global statistics.
    """
    detail_energies = []
    contrasts = []

    for img in image_list:
        img = load_volume(img).get_fdata()
        if mask_zero:
            img = img[img != 0]
        if img.size == 0:
            continue
        detail_energy = np.mean(np.abs(laplace(img)))
        contrast = np.std(img)

        detail_energies.append(detail_energy)
        contrasts.append(contrast)

    # trim to remove 5% of the highest and lowest values
    detail_energy_mean = trim_mean(detail_energies,trim_proportion)
    contrast_mean = trim_mean(contrasts,trim_proportion)

    # Normalize so typical energy/contrast → multiplier * energy ≈ 1.0
    sigma_multiplier = 1.0 / detail_energy_mean if detail_energy_mean > 0 else 1.0
    strength_multiplier = 1.0 / contrast_mean if contrast_mean > 0 else 1.0

    stats = {
        'detail_energy_mean': detail_energy_mean,
        'detail_energy_max': np.max(detail_energies) if detail_energies else 0,
        'contrast_mean': contrast_mean,
        'contrast_max': np.max(contrasts) if contrasts else 0,
        'sigma_multiplier': sigma_multiplier,
        'strength_multiplier': strength_multiplier
    }

    return sigma_multiplier, strength_multiplier, stats


def compute_sigma_strength_from_neighbors(
    pre_img, post_img, sigma_bounds=(0.5, 2.0), strength_bounds=(0.5, 2.0),
    sigma_multiplier=1.0, strength_multiplier=1.0
):
    """
    Computes adaptive sigma and strength from neighboring slices using global scaling multipliers.

    Parameters:
        pre_img (ndarray): Previous slice image.
        post_img (ndarray): Next slice image.
        sigma_bounds (tuple): Min/max bounds for sigma.
        strength_bounds (tuple): Min/max bounds for strength.
        sigma_multiplier (float): Multiplier from dataset for sigma.
        strength_multiplier (float): Multiplier from dataset for sharpening strength.

    Returns:
        sigma (float): Gaussian blur sigma.
        strength (float): Sharpening strength.
    """
    def local_detail_energy(img):
        return np.mean(np.abs(laplace(img)))

    def local_contrast(img):
        return np.std(img)

    # Average detail and contrast from neighbors
    detail_energy = np.mean([local_detail_energy(pre_img), local_detail_energy(post_img)])
    contrast = np.mean([local_contrast(pre_img), local_contrast(post_img)])

    # Apply scaling multipliers
    sigma = np.clip(sigma_multiplier * detail_energy, *sigma_bounds)
    strength = np.clip(strength_multiplier * contrast, *strength_bounds)

    return sigma, strength


def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = gaussian_filter(image, sigma)
    sharpened = image + strength * (image - blurred)
    return sharpened


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
def coreg_single_slice_orig(idx, output_dir, subject, img, all_image_fnames, template, 
                       target_slice_offset_list=[-1, -2, -3], zfill_num=4, 
                       input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl',
                       run_syn=True, run_rigid=True, previous_target_tag=None, 
                       scaling_factor=64, image_weights=None, retain_reg_mappings=False,
                       mask_zero=False, include_stack_template=True,regularization='Medium',
                       voxel_res=None, use_resolution_in_registration=False):
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
        include_stack_template (bool): If True, we also include the entire stack template with the same weight as the slice-specific template
            - no effect when only a single template is used
        voxel_res (tuple or list, optional): Voxel resolution in microns. Can be [x, y] for 2D images or [x, y, z] for 3D images (default is None).
            Note: voxel_res is used for setting output image metadata.
        use_resolution_in_registration (bool, optional): If True, registration uses physical resolution (ignore_res=False).
            If False (default), registration works in voxel space for better empirical performance.
    """
    all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below

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
        #generate the name of the template from the entire stack
        targets = [template[idx]]
        if include_stack_template: #we want to incldue the stack template to help reduce drift 
            stack_template = os.path.dirname(template[idx]) + os.path.sep + os.path.basename(template[idx]).split('_')[0] + template[idx].split('pix')[1]
            #now we append the stack appropriately
            sources.append(nifti)
            targets.append(stack_template)
            image_weights_ordered.append(image_weights[0])
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

    # we generate a unique temporary directory since ANTs may overwrite existing files
    # unique filenames *should* work, but we play it safe here

    with tempfile.TemporaryDirectory(prefix=f"coreg_slice_{idx}_") as tmp_output_dir:
        # logging.warning(f'Creating temporary slice directory {tmp_output_dir}')
        #original iterations:
        #                rigid_iterations=1000,
        #                coarse_iterations=2000,
        #                medium_iterations=1000, 
        #                fine_iterations=200,
    
        # Determine ignore parameters based on use_resolution_in_registration
        if use_resolution_in_registration:
            ignore_res = False
            ignore_affine = False  # Must respect affines when using resolution
        else:
            ignore_res = True  # Default: work in voxel space
            ignore_affine = False  # Still respect affines for resolution metadata
        
        with working_directory(tmp_output_dir):
            # Configure registration based on use_resolution_in_registration setting
            coreg_output = nighres.registration.embedded_antspy_2d_multi(
                source_images=sources,
                target_images=targets,
                image_weights=image_weights_ordered,
                run_rigid=run_rigid,
                rigid_iterations=5000,
                run_affine=False,
                run_syn=run_syn,
                coarse_iterations=2000,
                medium_iterations=1000, 
                fine_iterations=200,  #500 was a bit too aagro
                scaling_factor=scaling_factor,
                cost_function='Mattes', #MutualInformation
                interpolation='Linear',
                regularization=regularization,
                convergence=1e-6,
                mask_zero=mask_zero,
                ignore_affine=ignore_affine,
                ignore_orient=True, 
                ignore_res=ignore_res,
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
        logging.warning(f"\t\tRegistration completed for slice {idx}. \t\t(intermediate files deleted)")
    else:
        logging.warning(f"\t\tRegistration completed for slice {idx}. \t\t(intermediate files retained)")

def run_parallel_coregistrations(output_dir, subject, all_image_fnames, template, max_workers=3, 
                                  target_slice_offset_list=[-1,-2,-3], zfill_num=4, input_source_file_tag='coreg0nl', 
                                  reg_level_tag='coreg1nl', run_syn=True, run_rigid=True, previous_target_tag=None, 
                                  scaling_factor=64, image_weights=None, retain_reg_mappings=False, mask_zero=False,
                                  regularization='Medium', voxel_res=None, use_resolution_in_registration=False):
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
        voxel_res (tuple, optional): Voxel resolution in [x, y, z] format in microns (default is None).
        use_resolution_in_registration (bool, optional): If True, use physical resolution in registration (default is False).
    """

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, img in enumerate(all_image_fnames):
            futures.append(
                executor.submit(coreg_single_slice_orig, idx, output_dir, subject, img, all_image_fnames, template, 
                                target_slice_offset_list=target_slice_offset_list, zfill_num=zfill_num, 
                                input_source_file_tag=input_source_file_tag, reg_level_tag=reg_level_tag,
                                run_syn=run_syn, run_rigid=run_rigid, previous_target_tag=previous_target_tag,
                                scaling_factor=scaling_factor, image_weights=image_weights,
                                retain_reg_mappings=retain_reg_mappings,mask_zero=mask_zero, 
                                regularization=regularization, voxel_res=voxel_res,
                                use_resolution_in_registration=use_resolution_in_registration)
            )
        for future in as_completed(futures):
            try:
                future.result()
                # logging.warning("Registration completed for one slice.")
            except Exception as e:
                logging.error(f"Registration failed with error: {e}")

def run_cascading_coregistrations(output_dir, subject, all_image_fnames, anchor_slice_idx = None,
                                  missing_idxs_to_fill = None, zfill_num=4, input_source_file_tag='coreg0nl', 
                                  reg_level_tag='coreg1nl', previous_target_tag=None, run_syn=True, scaling_factor=64,
                                  voxel_res=None, use_resolution_in_registration=False):
    """
    Cascading slice-based registration

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
        
    voxel_res (tuple, optional): Voxel resolution in [x, y, z] format in microns (default is None).

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

    We also take care of all intermediate files by keeping them in a directory in /tmp/cascade_* that is cleaned
    on completion of the registration
    """
    # params for regularization of nonlinear deformations w/ 'SyNOnly'
        # from nighres, the last two numbers of the syn_param correspond to the flow and total sigmas (fluid and elastic deformations, respectively)
        # if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        # elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        # elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
    syn_flow_sigma = 4.0 #3 is the default w/ this ants.registration call
    syn_total_sigma = 3.0 #0 is the default w/ this ants.registration call

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
        valid_slices = set(range(len(all_image_fnames))) - set(missing_idxs_to_fill)
        closest_slice = min(valid_slices, key=lambda x: abs(x - anchor_slice_idx), default=None)
        if closest_slice is None:
            raise ValueError("No valid start slice index found in the stack.")
        anchor_slice_idx = closest_slice
    
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
        # first, we setup a temporary directory that cleans itself on completion
        with tempfile.TemporaryDirectory(prefix=f"cascade_reg_{idx}") as tmp_output_dir:
            with working_directory(tmp_output_dir): #make any files that would go to cwd are put in temp
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

                pre_to_post_rigid = ants.registration(fixed=target_img, moving=source_img, type_of_transform='Rigid', outprefix=tmp_output_dir+"/") #run rigid
                pre_aligned = ants.apply_transforms(fixed=target_img, moving=source_img, transformlist=pre_to_post_rigid['fwdtransforms']) #apply rigid
                
                # reg_aligned = do_reg([source], [target], file_name=output, output_dir=output_dir, run_syn=run_syn, scaling_factor=scaling_factor)
                # save_volume(output, load_volume(reg_aligned['transformed_source']) ,overwrite_file=True)
                
                if run_syn:
                    pre_to_post_nonlin = ants.registration(fixed=target_img, moving=pre_aligned, 
                                                        type_of_transform='SyNOnly',
                                                        initial_transform='Identity',
                                                        flow_sigma=syn_flow_sigma,
                                                        total_sigma=syn_total_sigma,
                                                        outprefix=tmp_output_dir+"/")
                    warpedmovout = pre_to_post_nonlin['warpedmovout']

                    ants.image_write(warpedmovout, output)
                else:
                    ants.image_write(pre_aligned, output) #if we are not running the nonlin, just write the rigidly aligned image
        logging.warning(f"\t\tCascade registration completed for slice {src_idxs[idx]}.")

def run_cascading_coregistrations_v2(output_dir, subject, all_image_fnames, anchor_slice_idx = None,
                                  missing_idxs_to_fill = None, zfill_num=4, input_source_file_tag='coreg0nl', 
                                  reg_level_tag='coreg1nl', previous_target_tag=None, run_syn=True, scaling_factor=64,
                                  mask_zero=False, voxel_res=None, use_resolution_in_registration=False):

    #TODO: some filenames are messed up due to ants automatic filenaming of outputs

    if previous_target_tag is not None:
        previous_tail = f'_{previous_target_tag}_ants-def0.nii.gz' #if we want to use the previous iteration rather than building from scratch every time (useful for windowing)
    else:
        previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'

    #identify a central slice to start our registration from, rather than anchoring @ the end
    #but we make sure that it is not a missing slice
    
    if anchor_slice_idx is None:
        anchor_slice_idx = int(numpy.floor(len(all_image_fnames)/2))

    if missing_idxs_to_fill is not None:
        valid_slices = set(range(len(all_image_fnames))) - set(missing_idxs_to_fill)
        closest_slice = min(valid_slices, key=lambda x: abs(x - anchor_slice_idx), default=None)
        if closest_slice is None:
            raise ValueError("No valid start slice index found in the stack.")
        anchor_slice_idx = closest_slice
    
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
        with tempfile.TemporaryDirectory(prefix=f'cascade_reg_v2_{idx}') as temp_out_dir:
            # img_basename = os.path.basename(all_image_fnames[idx]).split('.')[0]
            target_idx = trg_idxs[idx]
            moving_idx = src_idxs[idx]

            # in each case, only one source and one target, but we use the same code as above
            source = all_image_fnames_nii[moving_idx]
            target = all_image_fnames_new[target_idx] #targets always come from the new list, since this is where the registrered sources will be (and we pre-filled the start_slice_idx image)
            output = all_image_fnames_new[moving_idx]
            # previously was just do_reg()
            reg_aligned = do_reg_ants([source], [target], file_name=output, output_dir=temp_out_dir, run_syn=run_syn, 
                                scaling_factor=scaling_factor,mask_zero=mask_zero, voxel_res=voxel_res)
            save_volume(output, load_volume(reg_aligned['transformed_source']) ,overwrite_file=True)
            logging.warning(f"\t\tCascade registration version 2 completed for slice {src_idxs[idx]}.")

def compute_intermediate_non_linear_slice(pre_img, post_img, current_img=None, additional_coreg_mean = True, 
                                          idx=None, voxel_res=None):
    """
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
        
    voxel_res (tuple, optional): Voxel resolution in [x, y, z] format in microns (default is None).

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
    pre_to_post_nonlin = ants.registration(fixed=post_aligned, moving=pre_aligned, type_of_transform='SyNOnly',
                                           initial_transform='Identity') #must select identity as initial transform since we already did rigid
    post_to_pre_nonlin = ants.registration(fixed=pre_aligned, moving=post_aligned, type_of_transform='SyNOnly',
                                           initial_transform='Identity')

    # Step 4: Load the non-linear deformation fields as images
    # https://antspy.readthedocs.io/en/latest/registration.html #for reference
    pre_to_post_field = ants.image_read(pre_to_post_nonlin['fwdtransforms'][0]) # 2nd on the stack is the Affine
    post_to_pre_field = ants.image_read(post_to_pre_nonlin['invtransforms'][1]) # grab the inverse deformation field, which is the 2nd on the stack for inverted


    # Step 5: Convert the deformation fields to NumPy arrays and average them
    avg_field_data = (pre_to_post_field.numpy() + post_to_pre_field.numpy()) / 2
    avg_field = ants.from_numpy(avg_field_data, spacing=pre_to_post_field.spacing+(1.0,)) #need a 3rd dimension for spacing
    
    #we need to have the transform as a file, so we create a temp version here
    with tempfile.NamedTemporaryFile(prefix='XXX',suffix='.nii.gz') as temp_file:
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
        pre_to_post_nonlin = ants.registration(fixed=intermediate_img, moving=pre_aligned, type_of_transform='SyNOnly',
                                               initial_transform='Identity')
        post_to_pre_nonlin = ants.registration(fixed=intermediate_img, moving=post_aligned, type_of_transform='SyNOnly',
                                               initial_transform='Identity')

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
            current_to_template_nonlin = ants.registration(fixed=new_image,moving=current_aligned_rigid,type_of_transform='SyNOnly',
                                                           initial_transform='Identity')
            new_intermediate_img = current_to_template_nonlin['warpedmovout']
                
            intermediate_img_np = new_intermediate_img.numpy()

    if idx is not None: #if we passed an index value, this is to keep track of parallel so we pass it back
        return idx, intermediate_img_np
    else: 
        return intermediate_img_np


from contextlib import contextmanager

@contextmanager
def working_directory(path):
    """ 
        Helper function to change our current working directory to 
        ensure that temporary files created by ANTs are saved in the 
        correct directory, as specified. 

        ANTs will save temporary files in the current working directory under some conditions :(
    """
    prev_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

# same, with temporary files
def do_reg(sources, targets, run_rigid=True, run_syn=False, file_name='XXX', output_dir='./', scaling_factor=64, mask_zero=False, voxel_res=None, use_resolution_in_registration=False):
    """
    Helper function to perform registration between source and target images using ANTsPy w/ nighres
            course_iterations=100,
        medium_iterations=100,
        fine_iterations=50,
    
    Parameters:
        voxel_res (tuple, optional): Voxel resolution in [x, y, z] format (default is None).
            Note: voxel_res is used for setting output image metadata. 
        use_resolution_in_registration (bool, optional): If True, registration uses physical resolution (ignore_res=False).
            If False (default), registration works in voxel space (ignore_res=True) for better empirical performance.
            When True, ignore_affine is also set to False to respect affine transformations.
    """
    # Determine ignore parameters based on use_resolution_in_registration
    if use_resolution_in_registration:
        ignore_res = False
        ignore_affine = False  # Must respect affines when using resolution
    else:
        ignore_res = True  # Default: work in voxel space
        ignore_affine = False  # Still respect affines for resolution metadata
    
    with working_directory(output_dir):
        # Configure registration based on use_resolution_in_registration setting
        reg = nighres.registration.embedded_antspy_2d_multi(
            source_images=sources,
            target_images=targets,
            run_rigid=run_rigid,
            run_affine=False,
            run_syn=run_syn,
            scaling_factor=scaling_factor,
            cost_function='MutualInformation',
            interpolation='Linear',
            regularization='High',
            convergence=1e-6,
            mask_zero=mask_zero,
            ignore_affine=ignore_affine,
            ignore_orient=True, 
            ignore_res=ignore_res,
            save_data=True, 
            overwrite=True,
            file_name=file_name, 
            output_dir=output_dir
        )
    return reg


def do_reg_ants(sources, targets, run_rigid=True, run_syn=False,
                file_name='reg', output_dir='./', scaling_factor=64,
                mask_zero=False, syn_flow_sigma=3, syn_total_sigma=0, voxel_res=None):

    """
    Perform registration between source and target images using ANTsPy,
    mimicking the manual pipeline: rigid first, then SyN (optional).
    
    This function should always be wrapped by a function that generates a random directory and
    sets the cwd to it to ensure that there is no cross-talk between files if parallelized
    
    Parameters:
        sources, targets: list of image file paths (length 1 each expected)
        run_rigid: if True, perform rigid registration
        run_syn: if True, perform SyN registration after rigid
        file_name: base name for output files
        output_dir: where to write files
        scaling_factor, mask_zero: for compatibility (currently not used directly here)
        syn_flow_sigma, syn_total_sigma: optional SyN smoothing parameters
        voxel_res (tuple, optional): Voxel resolution in [x, y, z] format (default is None).

    Returns:
        dict with keys: 'transformed_source', 'rigid_transform', 'syn_transform'
    """
    import ants

    assert len(sources) == 1 and len(targets) == 1, "Only single source/target expected."

    source = sources[0]
    target = targets[0]
    # os.makedirs(output_dir, exist_ok=True) 

    with working_directory(output_dir):
        logging.info(f"Starting ANTs registration for: {os.path.basename(source)} → {os.path.basename(target)}")

        source_img = ants.image_read(source)
        target_img = ants.image_read(target)

        transformed_source = source_img  # default to identity if no registration

        rigid_transform = None
        syn_transform = None

        # Step 1: Rigid Registration
        if run_rigid:
            rigid_reg = ants.registration(
                fixed=target_img,
                moving=source_img,
                type_of_transform='Rigid',
                outprefix=file_name + '_rigid_'
            )
            rigid_transform = rigid_reg['fwdtransforms']
            transformed_source = ants.apply_transforms(
                fixed=target_img,
                moving=source_img,
                transformlist=rigid_transform
            )
        else:
            logging.info("Skipping rigid registration.")

        # Step 2: SyN Registration (optional)
        if run_syn:
            syn_reg = ants.registration(
                fixed=target_img,
                moving=transformed_source,
                type_of_transform='SyNOnly',
                initial_transform='Identity',
                flow_sigma=syn_flow_sigma,
                total_sigma=syn_total_sigma,
                outprefix=file_name + '_syn_'
            )
            syn_transform = syn_reg['fwdtransforms']
            transformed_source = syn_reg['warpedmovout']
        else:
            logging.info("Skipping SyN registration.")

        # Save final transformed image
        final_output_path = os.path.join(output_dir, file_name + '_transformed.nii.gz')
        ants.image_write(transformed_source, final_output_path)

        logging.info(f"Registration complete. Output saved to: {final_output_path}")

        if run_syn:
            return {
                'transformed_source': final_output_path,
                'rigid_transform': rigid_transform,
                'syn_transform': syn_transform
            }
        else:
            return {
                'transformed_source': final_output_path,
            }

    
def do_initial_translation_reg(sources, targets, root_dir=None, file_name='XXX', slice_idx_str=None, scaling_factor=64, mask_zero=False, voxel_res=None, use_resolution_in_registration=False):
    """
    Helper function to perform registration between source and target images using ANTsPy w/ nighres
    
    Parameters:
        voxel_res (tuple, optional): Voxel resolution in [x, y, z] format (default is None).
        use_resolution_in_registration (bool, optional): If True, registration uses physical resolution (ignore_res=False).
            If False (default), registration works in voxel space for better empirical performance.
    Doing only the initial translation step
    """
    #with tempfile.TemporaryDirectory(prefix=f"init_translation_slice_{idx}_") as tmp_output_dir:
    initial_output_dir = os.path.join(root_dir,f"init_translation_slice_{slice_idx_str}/")
    logging.warning(initial_output_dir)
    clean_file_name = os.path.basename(file_name) #nighres (which is called by do_reg) ignores output_dir if given file name with full path
    #with working_directory(initial_output_dir):
    reg = do_reg(sources, targets, run_rigid=False, run_syn=False, file_name=clean_file_name, 
                 output_dir=initial_output_dir, scaling_factor=scaling_factor, mask_zero=mask_zero, voxel_res=voxel_res,
                 use_resolution_in_registration=use_resolution_in_registration)
                
                ## this is what we were doing previously
                #  nighres.registration.embedded_antspy_2d_multi,source_images=sources, 
                #     target_images=targets,
                #     run_rigid=False,
                #     run_affine=False,
                #     run_syn=False,
                #     scaling_factor=64,
                #     cost_function='MutualInformation',
                #     interpolation='Linear',
                #     regularization='High',
                #     convergence=1e-6,
                #     mask_zero=mask_zero,
                #     ignore_affine=False, ignore_orient=False, ignore_res=False,
                #     save_data=True, overwrite=False,
                #     file_name=output
    return reg

def compute_intermediate_slice(pre_img, post_img, current_img=None, idx=None, delete_intermediate_files=True, 
                               reg_refinement_iterations=7, output_dir=None ,scaling_factor=64, mask_zero=False,
                               sigma_multiplier=None,strength_multiplier=None,voxel_res=None):
    """
    Computes an interpolated slice between two input images (pre_img and post_img) using iterative refinement 
    through rigid and SyN-based registration. Optionally, registers a third image (current_img) to the computed 
    average. Intermediate registration files can be saved or cleaned up based on user preference.

    Parameters:
    -----------
    pre_img : str
        File path to the image preceding the current slice.
    post_img : str
        File path to the image following the current slice.
    current_img : str, optional
        File path to the current slice image to be refined against the computed average. If None, no refinement is performed. Default is None.
    idx : int, optional
        Index of the current slice in the processing sequence. Returned with the result if provided. Default is None.
    delete_intermediate_files : bool, optional
        If True, temporary files generated during the registration process are deleted after execution. 
        If False, files are retained in the specified output directory. Default is True.
    reg_refinement_iterations : int, optional
        Number of iterations to refine the computed average slice. Default is 10.
    output_dir : str, optional
        Directory where intermediate files are saved if delete_intermediate_files is False. Default is None.
        **Leave as None** unless you are testing, in some cases ANTs creates files in the cwd and they can be overwritten during parallelized calls.
    scaling_factor: int, optional
        Scaling factor for the image resolution during registration. Default is 64 but this will fail with low resolution images
    voxel_res (tuple, optional): Voxel resolution in [x, y, z] format in microns (default is None).

    Returns:
    --------
    numpy.ndarray or tuple
        If `idx` is None, returns the interpolated slice as a 3D NumPy array.
        If `idx` is provided, returns a tuple `(idx, interpolated_slice)` where `interpolated_slice` is the 3D NumPy array.

    Notes:
    ------
    - The function leverages `nighres.registration.embedded_antspy_2d_multi` for registration tasks.
    - Intermediate slices are iteratively refined by registering the input slices to the computed average and updating it.
    - Temporary files are managed to ensure efficient disk usage unless explicitly retained.
    - Now runs multiple blurring steps to progressively sharpen to address potential alignment issues
    """

    deleted_flag = False

    # Create a temporary directory for intermediate files
    if delete_intermediate_files or output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='intermediate_slice_')
    else:
        temp_dir = output_dir
    
    if temp_dir[-1] is not os.sep:
        temp_dir += os.sep

    img_pre = load_volume(pre_img)
    img_post = load_volume(post_img)

    avg_orig = (img_pre.get_fdata() + img_post.get_fdata())/2
    img_avg = nibabel.Nifti1Image(avg_orig, affine=img_pre.affine, header=img_pre.header, dtype=img_pre.get_data_dtype())

    avg_fname = os.path.join(temp_dir, 'avg.nii.gz')
    save_volume(avg_fname, img_avg, overwrite_file=True)

    try:
        
        pre_post = do_reg_ants([pre_img], [avg_fname], file_name='pre_post', output_dir=temp_dir, 
                          scaling_factor=scaling_factor,mask_zero=mask_zero, voxel_res=voxel_res)
        post_pre = do_reg_ants([post_img], [avg_fname], file_name='post_pre', output_dir=temp_dir, 
                          scaling_factor=scaling_factor,mask_zero=mask_zero, voxel_res=voxel_res)

        reg_pre = pre_post['transformed_source']
        reg_post = post_pre['transformed_source']

        # Compute averages in slice space
        img = load_volume(reg_pre)
        avg_post = (img.get_fdata() + load_volume(post_img).get_fdata()) / 2

        img = load_volume(reg_post)
        avg_pre = (img.get_fdata() + load_volume(pre_img).get_fdata()) / 2

        avg = (avg_pre + avg_post) / 2
        blur_scale = 1.0
        avg = gaussian_filter(avg,sigma=blur_scale)
        avg = nibabel.Nifti1Image(avg, affine=img.affine, header=img.header, dtype=img.get_data_dtype())

        avg_fname = os.path.join(temp_dir, 'avg.nii.gz')
        save_volume(avg_fname, avg, overwrite_file=True)

        blur_scales = numpy.linspace(0,1,reg_refinement_iterations-1)[::-1]
        blur_scales = numpy.append(blur_scales,0)
        
        if reg_refinement_iterations is not None:
            # Refinement loop       
            for refinement_iter in range(reg_refinement_iterations):
                
                pre_avg = do_reg_ants([pre_img], [avg_fname], file_name='pre_avg', run_syn=True, output_dir=temp_dir, 
                                scaling_factor=scaling_factor,mask_zero=mask_zero, voxel_res=voxel_res)
                post_avg = do_reg_ants([post_img], [avg_fname], file_name='post_avg', run_syn=True, output_dir=temp_dir, 
                                scaling_factor=scaling_factor,mask_zero=mask_zero, voxel_res=voxel_res)
                
                img1 = load_volume(pre_avg['transformed_source'])
                img2 = load_volume(post_avg['transformed_source'])

                avg = (img1.get_fdata() + img2.get_fdata()) / 2
                blur_scale = blur_scales[refinement_iter]
                if not(blur_scale==0):
                    avg = gaussian_filter(avg,sigma=blur_scale)
                
                # XXX if the coregs are not stable, can fix this by changing this here XXX
                # #on the last iteration, we do not compute the average but rather the first image in the average space to reduce overlap errors
                # if (current_img is None) and (refinement_iter == reg_refinement_iterations - 1):
                #     avg = img1.get_fdata()

                #if we are on the last refinement_iter, we can do some sharpening and histogram matching for the synthetic slice that 
                # we just created
                if refinement_iter == reg_refinement_iterations - 1:
                    # histogram matching
                    avg = compute_histogram_matched_slice(avg, img1.get_fdata(),img2.get_fdata())

                    # optionally resharpen data
                    if (sigma_multiplier is not None) and (strength_multiplier is not None):
                        sigma, strength = compute_sigma_strength_from_neighbors(img1.get_fdata(),img2.get_fdata(),
                                                                    sigma_multiplier=sigma_multiplier,strength_multiplier=strength_multiplier)
                        avg = unsharp_mask(avg, sigma=sigma, strength=strength)
                    #histogram match to nonzero data
                
                avg = nibabel.Nifti1Image(avg, affine=img1.affine, header=img1.header, dtype=img1.get_data_dtype())
                save_volume(avg_fname, avg, overwrite_file=True)
                
        # If current_img is provided, refine it to match the final average
        if current_img is not None:
            current_avg = nibabel.load(do_reg_ants([current_img], [avg_fname], file_name='current_avg', run_syn=True, 
                                              output_dir=temp_dir, voxel_res=voxel_res)['transformed_source'],mask_zero=mask_zero)
        else:
            current_avg = avg

        # Cleanup temporary files
        if delete_intermediate_files or output_dir is None:
            deleted_flag = True
            shutil.rmtree(temp_dir)

        # Return the result
        if idx is not None:
            return idx, current_avg.get_fdata()
        else:
            return current_avg.get_fdata()
        
    finally:
        logging.warning(f"Temporary files for slice interpolation were saved in: {temp_dir}")
        if deleted_flag:
            logging.warning("[temporary files deleted]")


def generate_missing_slices(missing_fnames_pre,missing_fnames_post,current_fnames=None,method='intermediate_nonlin_mean',
                            nonlin_interp_max_workers=1,scaling_factor=64,mask_zero=False,
                            sigma_multiplier=None, strength_multiplier=None, voxel_res=None):
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
        
    voxel_res (tuple, optional): Voxel resolution in [x, y, z] format in microns (default is None).

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

                    #previously, was compute_intermediate_non_linear_slice,
                futures.append(executor.submit(
                    compute_intermediate_slice,
                    pre_img=img_fname_pre,
                    post_img=img_fname_post,
                    current_img=img_fname_current,
                    idx=idx,scaling_factor=scaling_factor,
                    mask_zero=mask_zero,
                    sigma_multiplier=sigma_multiplier,
                    strength_multiplier=strength_multiplier,
                    voxel_res=voxel_res
                ))
            for future in as_completed(futures):
                try:
                    the_idx, the_slice = future.result()
                    the_idxs.append(the_idx)
                    the_slices.append(the_slice)
                    logging.warning(f'\t\tParallel slice generation completed for missing slice index: {the_idx}')
                except Exception as e:
                    logging.warning('Parallel slice generation failed: {e}')
                    logging.warning(img_fname_pre)
                    logging.warning(img_fname_post)
                    logging.warning(img_fname_current)
                    logging.warning(f'scaling_factor = {scaling_factor}')
                    logging.warning("=============== CHECK THAT YOUR scaling_factor IS APPROPRIATE FOR YOUR IMAGE RESOLUTION ===============")
        idxs_order = numpy.argsort(the_idxs)
        sorted_slices = [the_slices[i] for i in idxs_order]
        missing_slices_interpolated= numpy.stack(sorted_slices, axis=-1) #reorder based on the indices that were passed
    else:
        missing_slices_interpolated = None
    return missing_slices_interpolated

def groupwise_stack_optimization(output_dir, subject, all_image_fnames, 
                                reg_level_tag, iterations=5, zfill_num=4,
                                scaling_factor=64, voxel_res=None):
    """
    Jointly optimize all slices together to enforce smooth transitions.
    Converts ANTs warps to nighres-compatible deformation maps.
    """
    import ants
    import nibabel as nib
    
    logging.warning("=" * 80)
    logging.warning("Starting groupwise optimization")
    logging.warning("=" * 80)
    
    # Load all registered slices
    images = []
    for idx, img_fname in enumerate(all_image_fnames):
        img_name = os.path.basename(img_fname).split('.')[0]
        img_path = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_name}_{reg_level_tag}_ants-def0.nii.gz"
        images.append(ants.image_read(img_path))
    
    # Iteratively refine to mean template
    for iteration in range(iterations):
        logging.warning(f"Groupwise iteration {iteration+1}/{iterations}")
        
        # Compute current mean template
        mean_data = np.mean([img.numpy() for img in images], axis=0)
        mean_template = ants.from_numpy(mean_data)
        mean_template.set_spacing(images[0].spacing)
        mean_template.set_origin(images[0].origin)
        mean_template.set_direction(images[0].direction)
        
        # Register each slice to mean with VERY smooth constraints
        registered_images = []
        for idx, img in enumerate(images):
            img_name = os.path.basename(all_image_fnames[idx]).split('.')[0]
            
            logging.info(f"  Processing slice {idx}/{len(images)}")
            
            # Create temporary directory for this registration
            with tempfile.TemporaryDirectory(prefix=f"groupwise_slice_{idx}_iter{iteration}_") as tmp_dir:
                output_prefix = os.path.join(tmp_dir, "reg")
                
                # Registration with high smoothness
                reg = ants.registration(
                    fixed=mean_template,
                    moving=img,
                    type_of_transform='SyNOnly',
                    flow_sigma=10,
                    total_sigma=8,
                    grad_step=0.05,
                    reg_iterations=(100, 50, 25),
                    outprefix=output_prefix
                )
                
                registered_images.append(reg['warpedmovout'])
                
                # On the LAST iteration, save and convert the transforms
                if iteration == iterations - 1:
                    # Define final output names
                    final_output_def = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_name}_{reg_level_tag}_groupwise_ants-def0.nii.gz"
                    final_output_map = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_name}_{reg_level_tag}_groupwise_ants-map.nii.gz"
                    final_output_invmap = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_name}_{reg_level_tag}_groupwise_ants-invmap.nii.gz"
                    
                    # Save the warped image
                    ants.image_write(reg['warpedmovout'], final_output_def)
                    
                    # Convert ANTs displacement field to deformation field
                    fwd_warp = os.path.join(tmp_dir, "reg0Warp.nii.gz")
                    inv_warp = os.path.join(tmp_dir, "reg0InverseWarp.nii.gz")
                    
                    if os.path.exists(fwd_warp):
                        convert_ants_warp_to_deformation(fwd_warp, final_output_map, img)
                    else:
                        logging.warning(f"Forward warp not found for slice {idx}")
                    
                    if os.path.exists(inv_warp):
                        convert_ants_warp_to_deformation(inv_warp, final_output_invmap, mean_template)
                    else:
                        logging.warning(f"Inverse warp not found for slice {idx}")
        
        # Update images for next iteration
        images = registered_images
        logging.warning(f"Completed groupwise iteration {iteration+1}/{iterations}")
    
    logging.warning("=" * 80)
    logging.warning("Groupwise optimization complete")
    logging.warning(f"Saved {len(all_image_fnames)} deformation maps")
    logging.warning("=" * 80)
    
    return images


### XXX UNTESTED FUNCTIONS BELOW
def convert_ants_warp_to_deformation(warp_file, output_file, reference_image=None):
    """
    Convert ANTs displacement field (warp) to absolute deformation field (map).
    Handles both 2D (z=1) and 3D data.
    
    ANTs warp: displacement vectors (how much each voxel moves)
    Deformation map: absolute coordinates (where each voxel ends up)
    
    Parameters:
    -----------
    warp_file : str
        Path to ANTs warp file (displacement field)
    output_file : str
        Path to save deformation map
    reference_image : ants.ANTsImage, optional
        Reference image for spatial information (currently unused but kept for API compatibility)
    """
    import nibabel as nib
    
    # Load the displacement field
    warp_nib = nib.load(warp_file)
    displacement = warp_nib.get_fdata()
    
    # Get image dimensions
    shape = displacement.shape[:3]  # (x, y, z)
    
    logging.info(f"Converting warp with shape: {shape}, displacement shape: {displacement.shape}")
    
    # Determine if 2D or 3D based on z dimension
    is_2d = (shape[2] == 1)
    
    if is_2d:
        # For 2D: only create grid for x and y
        x_coords = np.arange(shape[0], dtype=np.float32)
        y_coords = np.arange(shape[1], dtype=np.float32)
        z_coords = np.zeros(1, dtype=np.float32)  # Single z slice
        
        # Create 2D meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Expand to include z dimension (all zeros for 2D)
        zz = np.zeros_like(xx)
        
        # Stack to create identity mapping: (x, y, 1, 3)
        # Note: displacement from ANTs should have shape (x, y, 1, 2) or (x, y, 1, 3)
        deformation = np.zeros((*shape, 3), dtype=np.float32)
        deformation[..., 0] = xx
        deformation[..., 1] = yy
        deformation[..., 2] = zz
        
        # Add displacement
        # ANTs 2D warp might be (x, y, 1, 2) - only x,y components
        if displacement.shape[-1] == 2:
            # Only x and y displacements
            deformation[..., 0] += displacement[..., 0, 0]
            deformation[..., 1] += displacement[..., 0, 1]
            # z stays at 0 (no displacement in z)
        elif displacement.shape[-1] == 3:
            # Full 3D displacement (but z should be ~0)
            deformation += displacement
        else:
            raise ValueError(f"Unexpected displacement shape: {displacement.shape}")
            
    else:
        # For 3D: create full 3D grid
        coords = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing='ij'
        )
        
        # Stack into deformation field: (x, y, z, 3)
        deformation = np.stack(coords, axis=-1)
        
        # Add displacement
        deformation += displacement
    
    # Save as deformation map
    deformation_nib = nib.Nifti1Image(
        deformation.astype(np.float32), 
        affine=warp_nib.affine, 
        header=warp_nib.header
    )
    nib.save(deformation_nib, output_file)
    
    logging.info(f"Converted {'2D' if is_2d else '3D'} warp to deformation map: {os.path.basename(output_file)}")
    logging.info(f"  Deformation field shape: {deformation.shape}")


def convert_ants_warp_to_deformation_v2(warp_file, output_file, reference_image):
    """
    Alternative version: squeeze out singleton z dimension for true 2D.
    Use this if nighres expects 2D maps to be (x, y, 2) not (x, y, 1, 3).
    """
    import nibabel as nib
    
    # Load the displacement field
    warp_nib = nib.load(warp_file)
    displacement = warp_nib.get_fdata()
    
    # Get image dimensions
    shape = displacement.shape[:3]
    is_2d = (shape[2] == 1)
    
    if is_2d:
        # Squeeze out z dimension for true 2D
        logging.info(f"Creating 2D deformation map (shape will be {shape[0]}, {shape[1]}, 2)")
        
        # Create 2D identity grid
        x_coords = np.arange(shape[0], dtype=np.float32)
        y_coords = np.arange(shape[1], dtype=np.float32)
        
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Create 2D deformation field (x, y, 2) - only x and y components
        deformation_2d = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
        deformation_2d[..., 0] = xx
        deformation_2d[..., 1] = yy
        
        # Add displacement (squeeze z dimension)
        if displacement.shape[-1] == 2:
            # displacement is (x, y, 1, 2)
            deformation_2d[..., 0] += displacement[..., 0, 0]
            deformation_2d[..., 1] += displacement[..., 0, 1]
        elif displacement.shape[-1] == 3:
            # displacement is (x, y, 1, 3) - take only x,y
            deformation_2d[..., 0] += displacement[..., 0, 0]
            deformation_2d[..., 1] += displacement[..., 0, 1]
        
        # Create 2D affine (remove z dimension)
        affine_2d = warp_nib.affine[:3, :3]  # 3x3 instead of 4x4
        
        # Save as 2D deformation map
        deformation_nib = nib.Nifti1Image(
            deformation_2d.astype(np.float32),
            affine=warp_nib.affine,  # Keep original affine for compatibility
            header=warp_nib.header
        )
        
    else:
        # 3D case - same as before
        coords = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing='ij'
        )
        deformation = np.stack(coords, axis=-1)
        deformation += displacement
        
        deformation_nib = nib.Nifti1Image(
            deformation.astype(np.float32),
            affine=warp_nib.affine,
            header=warp_nib.header
        )
    
    nib.save(deformation_nib, output_file)
    logging.info(f"Converted warp to deformation map: {os.path.basename(output_file)}")


def generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=4,reg_level_tag='coreg12nl',
                                per_slice_template=False,missing_idxs_to_fill=None, slice_template_type='median'
                                ,nonlin_interp_max_workers=1,scaling_factor=64,voxel_res=None,mask_zero=False,
                                sigma_multiplier=None, strength_multiplier=None, across_slice_smoothing_sigma=0,
                                match_histograms_to_slice=185):
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
        slice_template_type (str or list, optional): Specifies the method for generating the slice templates. Can be 'nochange', 'mean', 'median' (only one of these) 'nonlin', or a list of these methods (default is 'median').
                                                    'nochange' saves the deformed slice as the template
        nonlin_interp_max_workers (int, optional): Number of workers to use for parallelized non-linear interpolation (default is 1).
        scaling_factor (int, optional): Scaling factor for registrations (default is 64)
        voxel_res (tuple, optional): Voxel resolution of the images (default is None).
        mask_zero (bool, optional): If True, masks zero values in the images (default is False).
        sigma_multiplier (float, optional): Multiplier for the Gaussian filter applied to the missing slice images after interpolation (default is None).
        strength_multiplier (float, optional): Multiplier for the unsharp mask applied to the missing slice images after interpolation (default is None).
        across_slice_smoothing_sigma (float, optional): Sigma value for Gaussian smoothing applied across slices (default is None, 5 is reasonable).
        match_histograms_to_slice (int, optional): Slice index to match histograms to (default is 185) for output only, not used in 
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
                - `'nochange'`: The deformed slice itself as the template
            
        5. **Saving Slice Templates**:
            - If `per_slice_template` is set to True, templates for each slice are saved individually, both for median and non-linear methods, as applicable.

        6. **Return Value**:
            - The function returns the filename(s) of the generated templates. If individual slice templates are generated, a list of template filenames is returned; otherwise, a single template filename is returned.
    """
    
    #we can also output a per_slice_template based on the median of the current and neighbouring slices
    if across_slice_smoothing_sigma is None:
        across_slice_smoothing_sigma = 0
    
    if voxel_res is None:
        voxel_res = [1.0,1.0,1.0]

    stack = []
    stack_tail = f'_{reg_level_tag}_stack.nii.gz'
    img_stack = output_dir+subject+stack_tail
    img_stack_nonlin = img_stack.replace('.nii.gz','_nonlin.nii.gz')
    template_tail = f'_{reg_level_tag}_template.nii.gz'
    template_nonlin_tail = f'_{reg_level_tag}_template_nonlin.nii.gz'
    template_nochange_tail = f'_{reg_level_tag}_template_nochange.nii.gz'

    template = output_dir+subject+template_tail

    template_list = []
    template_nonlin_list = []

    img_tail = f'_{reg_level_tag}_ants-def0.nii.gz'

    if (os.path.isfile(img_stack)):
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
                                                                  nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                                  scaling_factor=scaling_factor,mask_zero=mask_zero,
                                                                  sigma_multiplier=sigma_multiplier, 
                                                                  strength_multiplier=strength_multiplier,
                                                                  voxel_res=voxel_res)

            
            #now we can fill the slices with the interpolated value
            for idx,missing_idx in enumerate(missing_idxs_to_fill):
                if not os.path.exists(output_dir + "missing_slices/"):
                    os.makedirs(output_dir + "missing_slices/")
                
                #we overwrite the missing slices with the interpolated values and insert it into the img stack
                missing_fname = missing_fnames_current[idx]
                #but first we copy to our missing slices directory
                shutil.copy(missing_fname,output_dir + "missing_slices/")
                interp_slice = missing_slices_interpolated[...,idx]
                header = nibabel.Nifti1Header()
                header.set_data_shape(interp_slice.shape)
                
                affine = create_affine(interp_slice.shape, voxel_res=voxel_res)
                
                nifti = nibabel.Nifti1Image(interp_slice,affine=affine,header=header)
                nifti.update_header()
                save_volume(missing_fname,nifti)
                shutil.copy(missing_fname, output_dir + f"missing_slices/{os.path.basename(missing_fname).split('.')[0]}_synthetic_slice.nii.gz")
                # print(idx)
                # print(missing_idx)
                # print(numpy.shape(missing_slices_interpolated))
                # print(numpy.shape(img))
                img[...,missing_idx] = interp_slice

        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        affine = create_affine(img.shape, voxel_res=voxel_res)
        
        if (across_slice_smoothing_sigma is not None) and (across_slice_smoothing_sigma > 0):
            # save the original stack first
            nifti = nibabel.Nifti1Image(img,affine=affine,header=header)
            save_volume(img_stack.split('_stack.nii.gz')[0]+'_orig_stack.nii.gz',nifti)
            img = gaussian_filter(img,sigma=(0,0,across_slice_smoothing_sigma)) #apply 1d smoothing
        elif across_slice_smoothing_sigma<0:
            logging.warning('Smoothing sigma is negative, skipping across slice smoothing') 

        if match_histograms_to_slice is not None and match_histograms_to_slice >0:
            img_matched = np.zeros_like(img)
            for hist_slice_idx in range(img.shape[-1]):
                img_matched[...,hist_slice_idx] = compute_histogram_matched_slice(img[...,hist_slice_idx],img[...,match_histograms_to_slice])
            nifti = nibabel.Nifti1Image(img_matched,affine=affine,header=header)
            save_volume(img_stack.split('_stack.nii.gz')[0]+'_histMatched_stack.nii.gz',nifti)

        nifti = nibabel.Nifti1Image(img,affine=affine,header=header)
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
                    # elif missing_idxs_to_fill is not None and idx in missing_idxs_to_fill:
                    #     logging.warning(f'========> NOT COMPUTING THE MEDIAN FOR MISSING SLICE at idx {idx}')
                    #     slice_template = img[...,idx]
                    else: #take one on each side and the current slice
                        start = idx-1
                        stop = idx+2
                        slice_template = numpy.median(img[...,start:stop],axis=-1)
            
                    header.set_data_shape(slice_template.shape)
                    affine = create_affine(slice_template.shape, voxel_res=voxel_res)
                    nifti = nibabel.Nifti1Image(slice_template,affine=affine,header=header)
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
                    affine = create_affine(slice_template.shape, voxel_res=voxel_res)
                    nifti = nibabel.Nifti1Image(slice_template,affine=affine,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname,nifti)
                    template_list.append(slice_template_fname)      
            
            elif 'nochange' in slice_template_type:
                logging.warning('Generating nochange slice templates')
                for idx,img_name in enumerate(all_image_fnames):
                    img_name = os.path.basename(img_name).split('.')[0]
                    slice_template_fname = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_nochange_tail
                    slice_template = img[...,idx]
                    header.set_data_shape(slice_template.shape)
                    affine = create_affine(slice_template.shape, voxel_res=voxel_res)
                    nifti = nibabel.Nifti1Image(slice_template,affine=affine,header=header)
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
                    affine = create_affine(slice_template.shape, voxel_res=voxel_res)
                    nifti = nibabel.Nifti1Image(slice_template,affine=affine,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname,nifti)
                    template_nonlin_list.append(slice_template_fname)
                
                # then use these images to generate the interpolations, skipping the first and last images (as anchors) 
                missing_fnames_pre_1 = template_nonlin_list[:-2] #starting from 0, skip the last two (b/c last has no pair), treated as pre
                missing_fnames_post_1 = template_nonlin_list[2:] #starting from 1 (skip the first one), treated as pre
                interp_template_slices = generate_missing_slices(missing_fnames_pre_1,missing_fnames_post_1,
                                                                 current_fnames=template_nonlin_list[1:-1],
                                                                 method='intermediate_nonlin_mean',
                                                                 nonlin_interp_max_workers=nonlin_interp_max_workers,
                                                                 scaling_factor=scaling_factor,mask_zero=mask_zero,
                                                                 voxel_res=voxel_res)

                #fill the image stack with the interpolated slices
                # save with a differnt fname so that we can see what this looks like
                img[...,1:-1] = interp_template_slices
                nifti = nibabel.Nifti1Image(img,affine=affine,header=header)
                save_volume(img_stack_nonlin,nifti)

                #save them as their own individual templates, saving over the original ones that were not yet interpolated
                for idx,slice_template_fname_nonlin in enumerate(template_nonlin_list):
                    slice_template = img[...,idx]
                    header.set_data_shape(slice_template.shape)
                    affine = create_affine(slice_template.shape, voxel_res=voxel_res)
                    nifti = nibabel.Nifti1Image(slice_template,affine=affine,header=header)
                    nifti.update_header()
                    save_volume(slice_template_fname_nonlin,nifti)
            
        #now save the single template (as a median only)
        img = numpy.median(img,axis=2)
        nifti = nibabel.Nifti1Image(img,affine=affine,header=header)
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
    logging.warning(f'Removing old intermediate files for idx {idx}: {img_name}')
    for f in glob.glob(output_dir + subject + f'_{str(idx).zfill(zfill_num)}_' + img_name + "*_ants-*map.nii.gz"):
        if out_tail not in f:
            os.remove(f)
            time.sleep(0.5)
    logging.warning(f'---> Intermediate files removed')

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
    logging.warning(all_image_fnames)    
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
                          zfill_num=4,reg_level_tag1='coreg1nl', reg_level_tag2='coreg2nl',reg_output_tag='coreg12nl',per_slice_template=False,
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
            
def create_affine(shape, voxel_res=None, center=True):
    """
    Creates an affine transformation matrix. Should be RAS+
    
    Parameters:
    - shape (tuple): Shape of the image.
    - voxel_res (tuple or list, optional): Voxel resolution [x, y, z].
    - center (bool): If True, center the image at the origin.
    
    Returns:
    - numpy.ndarray: 4x4 affine matrix.
    """
    if voxel_res is None:
        voxel_res = [1.0, 1.0, 1.0]
    
    voxel_res = list(voxel_res)
    while len(voxel_res) < 3:
        voxel_res.append(1.0)
    
    affine = numpy.eye(4)
    affine[0, 0] = voxel_res[0]
    affine[1, 1] = voxel_res[1]
    affine[2, 2] = voxel_res[2]
    
    if center:
        # Center at origin
        affine[0, 3] = -(shape[0] - 1) / 2.0 * voxel_res[0]
        affine[1, 3] = -(shape[1] - 1) / 2.0 * voxel_res[1]
        if len(shape) >= 3 and shape[2] > 1:
            affine[2, 3] = -(shape[2] - 1) / 2.0 * voxel_res[2]
    
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
