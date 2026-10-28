"""Opt-in extensions to slice_registration_functions, kept separate so that the original functions are unchanged.

coreg_single_slice_v3 / run_parallel_coregistrations_v3
    Same as coreg_single_slice_orig / run_parallel_coregistrations, plus two options.

    `stack_template_list` / `stack_template_weight_scale`:
    per-slice files that replace the whole-stack median target (e.g. MRI planes from mri_anchor.py),
    and a factor for the weight of that target. None / 1.0 (default) is the original behaviour.

    `neighbour_target_tag`:
    the registration level whose deformed slices (`_{tag}_ants-def0.nii.gz`) are used as the neighbour targets.
    - None (default): neighbours come from the source level (input_source_file_tag / previous_target_tag),
      identical to the original functions.
    - e.g. 'coreg12nl_win12_rigsyn_0': neighbours come from the previous STAGE1 output, while the source
      still comes from input_source_file_tag (e.g. 'coreg0nl'), so the transform chain is unchanged.
    Output file names and mapping formats are identical to the original functions.
"""
import re

from slice_registration_functions import *


def coreg_single_slice_v3(idx, output_dir, subject, img, all_image_fnames, template,
                          target_slice_offset_list=[-1, -2, -3], zfill_num=4,
                          input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl',
                          run_syn=True, run_rigid=True, previous_target_tag=None,
                          scaling_factor=64, image_weights=None, retain_reg_mappings=False,
                          mask_zero=False, include_stack_template=True, regularization='Medium',
                          voxel_res=None, use_resolution_in_registration=False,
                          cost_function='Mattes', default_value=0, neighbour_target_tag=None,
                          stack_template_list=None, stack_template_weight_scale=1.0):
    """
    coreg_single_slice_orig with a per-slice replacement for the stack template and a separate tag for the neighbour targets (see module docstring).
    All other parameters are as in coreg_single_slice_orig.
    """
    all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below

    img_basename = os.path.basename(img).split('.')[0]

    if previous_target_tag is not None:
        previous_tail = f'_{previous_target_tag}_ants-def0.nii.gz' #if we want to use the previous iteration rather than building from scratch every time (useful for windowing)
    else:
        previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'

    if neighbour_target_tag is not None:
        neighbour_tail = f'_{neighbour_target_tag}_ants-def0.nii.gz'
    else:
        neighbour_tail = previous_tail

    nifti = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_basename}{previous_tail}"
    sources = [nifti]
    image_weights_ordered = [image_weights[0]]

    # Assign the correct template for this slice
    if type(template) is list:
        #generate the name of the template from the entire stack
        targets = [template[idx]]
        if include_stack_template: #we want to incldue the stack template to help reduce drift
            if stack_template_list is not None:
                stack_template = stack_template_list[idx]
            else:
                stack_template = os.path.dirname(template[idx]) + os.path.sep + os.path.basename(template[idx]).split('_')[0] + template[idx].split('pix')[1]
            #now we append the stack appropriately
            sources.append(nifti)
            targets.append(stack_template)
            image_weights_ordered.append(image_weights[0] * stack_template_weight_scale)
    else:
        targets = [template]

    # Determine the sources and targets based on `target_slice_offset_list`
    for idx2, slice_offset in enumerate(target_slice_offset_list):
        if slice_offset < 0 and idx >= abs(slice_offset):
            prev_nifti = f"{output_dir}{subject}_{str(idx + slice_offset).zfill(zfill_num)}_{all_image_names[idx + slice_offset]}{neighbour_tail}"
            sources.append(nifti)
            targets.append(prev_nifti)
            image_weights_ordered.append(image_weights[idx2 + 1])
        elif slice_offset > 0 and idx < len(all_image_names) - slice_offset:
            next_nifti = f"{output_dir}{subject}_{str(idx + slice_offset).zfill(zfill_num)}_{all_image_names[idx + slice_offset]}{neighbour_tail}"
            sources.append(nifti)
            targets.append(next_nifti)
            image_weights_ordered.append(image_weights[idx2 + 1])

    logging.info(f'\n\tslice_idx: {idx}\n\t\tsources: {sources[0].split("/")[-1]}\n\t\ttargets: {[t.split("/")[-1] for t in targets]}\n\t\tweights: {image_weights_ordered}') #source is always the same

    output = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_basename}_{reg_level_tag}"

    with tempfile.TemporaryDirectory(prefix=f"coreg_slice_{idx}_") as tmp_output_dir:
        if use_resolution_in_registration:
            ignore_res = False
            ignore_affine = False  # Must respect affines when using resolution
        else:
            ignore_res = True  # Default: work in voxel space
            ignore_affine = False  # Still respect affines for resolution metadata

        with working_directory(tmp_output_dir):
            coreg_output = embedded_antspy_2d_multi(
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
                cost_function=cost_function,
                interpolation='Linear',
                regularization=regularization,
                convergence=1e-6,
                mask_zero=mask_zero,
                ignore_affine=ignore_affine,
                ignore_orient=True,
                ignore_res=ignore_res,
                save_data=True,
                overwrite=False,
                file_name=output,
                default_value=default_value
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


def run_parallel_coregistrations_v3(output_dir, subject, all_image_fnames, template, max_workers=3,
                                    target_slice_offset_list=[-1,-2,-3], zfill_num=4, input_source_file_tag='coreg0nl',
                                    reg_level_tag='coreg1nl', run_syn=True, run_rigid=True, previous_target_tag=None,
                                    scaling_factor=64, image_weights=None, retain_reg_mappings=False, mask_zero=False,
                                    regularization='Medium', voxel_res=None, use_resolution_in_registration=False,
                                    cost_function='Mattes', default_value=0, neighbour_target_tag=None,
                                    stack_template_list=None, stack_template_weight_scale=1.0):
    """
    run_parallel_coregistrations using coreg_single_slice_v3 (see module docstring for the added options).
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, img in enumerate(all_image_fnames):
            futures.append(
                executor.submit(coreg_single_slice_v3, idx, output_dir, subject, img, all_image_fnames, template,
                                target_slice_offset_list=target_slice_offset_list, zfill_num=zfill_num,
                                input_source_file_tag=input_source_file_tag, reg_level_tag=reg_level_tag,
                                run_syn=run_syn, run_rigid=run_rigid, previous_target_tag=previous_target_tag,
                                scaling_factor=scaling_factor, image_weights=image_weights,
                                retain_reg_mappings=retain_reg_mappings,mask_zero=mask_zero,
                                regularization=regularization, voxel_res=voxel_res,
                                use_resolution_in_registration=use_resolution_in_registration,
                                cost_function=cost_function, default_value=default_value,
                                neighbour_target_tag=neighbour_target_tag,
                                stack_template_list=stack_template_list,
                                stack_template_weight_scale=stack_template_weight_scale)
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Registration failed with error: {e}")


# ---------------------------------------------------------------------------------------------------------------------
# groupwise_stack_optimization_embedded_antspy_v2
#     Copy of slice_registration_functions.groupwise_stack_optimization_embedded_antspy with one fix: the old
#     suffixes are removed with a regex. The original removes a fixed list of suffixes, so from iteration 11 on the
#     names lose part of '_groupwise_iter10' (e.g. 'rigsyn_40_groupwise_iter11') and the stack step cannot find them.
#     For 10 iterations or fewer the output is identical to the original.
# ---------------------------------------------------------------------------------------------------------------------
def groupwise_stack_optimization_embedded_antspy_v2(output_dir, subject, all_image_fnames, 
                                reg_level_tag, iterations=10, idxs_to_compute_mean='all', zfill_num=4,
                                scaling_factor=64, use_resolution_in_registration=True, max_workers=50,
                                weighted_mask_template=True, distance_sigma=3,
                                use_deformed_source_after_iteration=4,
                                local_template_window=5, use_local_template_after_iteration=3,
                                cost_function='MutualInformation'):
    """
    Iteratively register all slices to a mean template using nighres/ANTsPy SyN registration.
    
    This function performs groupwise optimization to enforce smooth slice-to-slice transitions.
    Each iteration: (1) computes a weighted mean template, (2) registers all slices to it in 
    parallel, and (3) updates images for the next iteration. Supports local per-slice templates
    for smoother results and switching from original to deformed source images mid-optimization.
    
    Transform Chain Logic:
        - Iterations 0 to (use_deformed_source_after_iteration-1): Register orig → template
          (maps not needed except the last one before switching)
        - Iterations use_deformed_source_after_iteration onward: Chain transforms
          (all maps retained to compose: orig → iterN-1 → iterN → ... → final)
    
    Parameters
    ----------
    output_dir : str
        Directory for output files (deformed images, maps, templates, stacks).
    subject : str
        Subject identifier prefix for output filenames.
    all_image_fnames : list of str
        Original image filenames (used to derive output naming).
    reg_level_tag : str
        Tag identifying input registration level (e.g., 'coreg12nl_win12_rigsyn_4').
        Input files expected at: {output_dir}{subject}_{idx}_{img_name}_{reg_level_tag}_ants-def0.nii.gz
    iterations : int, default=5
        Number of groupwise optimization iterations.
    idxs_to_compute_mean : 'all', int, list of int, or None, default='all'
        Which iterations recompute the mean template. 'all' = every iteration,
        None = only iteration 0, int/list = specific iterations (0 always included).
    zfill_num : int, default=4
        Zero-padding width for slice indices in filenames.
    scaling_factor : int, default=64
        Scaling factor passed to nighres registration.
    use_resolution_in_registration : bool, default=True
        If True, use physical resolution (ignore_res=False). If False, work in voxel space.
    max_workers : int, default=1
        Number of parallel workers for slice registration.
    weighted_mask_template : bool, default=True
        If True, weight template contributions by signed distance from tissue boundary
        (reduces edge artifacts). If False, use simple mean.
    distance_sigma : float, default=3
        Sigma for signed distance weighting (pixels). Controls boundary transition softness.
    use_deformed_source_after_iteration : int or None, default=3
        If None, always register from original images. If int N, switch to using
        deformed images as source starting at iteration N. Maps are retained from
        iteration (N-1) onward to enable transform composition.
    local_template_window : int or None, default=3
        If None, use global mean template for all slices. If int k, compute per-slice
        templates from ±k neighboring slices with Gaussian distance weighting.
    use_local_template_after_iteration : int or None, default=2
        If None, use local templates from iteration 0. If int N, use global template
        for iterations 0 to N-1, then switch to local templates.
    
    Returns
    -------
    list of nibabel.Nifti1Image
        Final registered images after all iterations.
    
    Output Files
    ------------
    Per iteration:
        - groupwise_iter{N}_template.nii.gz : Global mean template
        - groupwise_iter{N}_weights.nii.gz : Sum of distance weights (if weighted_mask_template)
        - groupwise_iter{N}_local_template_slice{idx}.nii.gz : Per-slice templates (if local)
    Per slice per iteration:
        - {img_name}_groupwise_iter{N}_ants-def0.nii.gz : Deformed image
        - {img_name}_groupwise_iter{N}_ants-map.nii.gz : Forward deformation map (if retained)
        - {img_name}_groupwise_iter{N}_ants-invmap.nii.gz : Inverse map (if retained)
    
    Notes
    -----
    - Uses 'VeryHigh' regularization in embedded_antspy_2d_multi for smooth deformations.
    - Maps are only retained for iterations where chaining is needed (1 previous to and all deformed source iterations)
      or the final iteration, to save disk space.
    """
    import copy
    import nibabel as nib
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np
    import os
    
    logging.warning("=" * 80)
    logging.warning("Starting groupwise optimization")
    logging.warning("=" * 80)
    
    if idxs_to_compute_mean is None:
        idxs_to_compute_mean = [0]
    elif idxs_to_compute_mean == 'all':
        idxs_to_compute_mean = list(range(iterations))
    else:
        if isinstance(idxs_to_compute_mean, int):
            idxs_to_compute_mean = [idxs_to_compute_mean]
        if 0 not in idxs_to_compute_mean:
            idxs_to_compute_mean.append(0)
            
    # Determine ignore parameters based on use_resolution_in_registration
    if use_resolution_in_registration:
        ignore_res = False
        ignore_affine = False  # Must respect affines when using resolution
    else:
        ignore_res = True  # Default: work in voxel space
        ignore_affine = False  # Still respect affines for resolution metadata
    
    # Load all registered slices
    images = []
    images_fnames = []
    for idx, img_fname in enumerate(all_image_fnames):
        img_name = os.path.basename(img_fname).split('.')[0]
        img_path = f"{output_dir}{subject}_{str(idx).zfill(zfill_num)}_{img_name}_{reg_level_tag}_ants-def0.nii.gz"
        images.append(nib.load(img_path))
        images_fnames.append(img_path)

    orig_images_fnames = copy.deepcopy(images_fnames)
    # Iteratively refine to mean template

    for iteration in range(iterations): 
        logging.warning(f"Groupwise iteration {iteration+1}/{iterations}")
        
        # Determine if we should use local templates this iteration
        
        use_local_this_iter = (local_template_window is not None and 
                               (use_local_template_after_iteration is None or 
                                iteration >= use_local_template_after_iteration))
        
        # Compute current mean template from CURRENT images
        # we need to weight inside the loop for each img
        # 
        
        # compute a shared mask to restrict registration to the tissue-containing areas, this can help with convergence and reduce expansion/contraction
        if iteration in idxs_to_compute_mean:
            logging.warning("  Computing mean template with weighted mask...")
            if weighted_mask_template:
                mean_data = np.zeros_like(images[0].get_fdata())
                sum_weights = np.zeros_like(images[0].get_fdata())
                for _img in images:
                    data = _img.get_fdata()
                    mask = generate_tissue_mask(data)
                    w_mask = compute_signed_distance_weight(mask, sigma=distance_sigma)
                    mean_data += w_mask * data
                    sum_weights += w_mask
                    
                mean_data = np.divide(mean_data, sum_weights, where=sum_weights > 1e-6, out=mean_data)
                weights_fname = os.path.join(output_dir, f"groupwise_iter{iteration}_weights.nii.gz")
                weights_template = nib.Nifti1Image(sum_weights, affine=images[0].affine, header=images[0].header)
                weights_template.to_filename(weights_fname)
            else:
                mean_data = np.mean([img.get_fdata() for img in images], axis=0)

            mean_template = nib.Nifti1Image(mean_data, affine=images[0].affine, header=images[0].header)

        # Save global template to disk (used for diagnostics and when not using local templates)
        template_fname = os.path.join(output_dir, f"groupwise_iter{iteration}_template.nii.gz")
        mean_template.to_filename(template_fname)
        
        # Compute local templates if needed
        template_fnames = []  # Will store per-slice template filenames
        if use_local_this_iter:
            logging.warning(f"  Computing LOCAL templates with window ±{local_template_window} slices...")
            n_slices = len(images)
            
            for slice_idx in range(n_slices):
                # Compute Gaussian weights for neighboring slices
                local_mean_data = np.zeros_like(images[0].get_fdata())
                local_sum_weights = np.zeros_like(images[0].get_fdata())
                
                # Define neighborhood range
                start_idx = max(0, slice_idx - local_template_window)
                end_idx = min(n_slices, slice_idx + local_template_window + 1)
                
                # Gaussian weighting based on slice distance (sigma = window/2 for smooth falloff)
                slice_sigma = local_template_window / 2.0
                
                for neighbor_idx in range(start_idx, end_idx):
                    # Gaussian weight based on distance from current slice
                    slice_distance = abs(neighbor_idx - slice_idx)
                    slice_weight = np.exp(-0.5 * (slice_distance / slice_sigma) ** 2)
                    
                    neighbor_data = images[neighbor_idx].get_fdata()
                    
                    if weighted_mask_template:
                        mask = generate_tissue_mask(neighbor_data)
                        spatial_weight = compute_signed_distance_weight(mask, sigma=distance_sigma)
                        combined_weight = slice_weight * spatial_weight
                    else:
                        combined_weight = slice_weight
                    
                    local_mean_data += combined_weight * neighbor_data
                    local_sum_weights += combined_weight
                
                # Normalize
                local_mean_data = np.divide(local_mean_data, local_sum_weights, 
                                           where=local_sum_weights > 1e-6, out=local_mean_data)
                
                # Save local template for this slice
                local_template_fname = os.path.join(output_dir, 
                    f"groupwise_iter{iteration}_local_template_slice{str(slice_idx).zfill(4)}.nii.gz")
                local_template = nib.Nifti1Image(local_mean_data, affine=images[0].affine, header=images[0].header)
                local_template.to_filename(local_template_fname)
                template_fnames.append(local_template_fname)
            
            logging.warning(f"  Created {len(template_fnames)} local templates")
        else:
            # Use global template for all slices
            template_fnames = [template_fname] * len(images)
        
        logging.warning(f"Registering {len(images_fnames)} slices to {'local' if use_local_this_iter else 'global'} template in parallel...")
        
        # Parallel registration of all slices
        results = [None] * len(images_fnames)
        
        # Determine which source files to use for this iteration
        if use_deformed_source_after_iteration is None or iteration < use_deformed_source_after_iteration:
            source_fnames = orig_images_fnames
            logging.warning(f"  Using ORIGINAL images as registration source")
        else:
            source_fnames = images_fnames
            logging.warning(f"  Using DEFORMED images from previous iteration as registration source")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Submit all groupwise registrations in parallel,
            for idx, full_img_name in enumerate(source_fnames):
                # Extract img_name for THIS slice
                img_name = os.path.basename(full_img_name).split('.')[0]
                # Remove any previous suffixes to avoid super long names
                # XXX TODO: this is a bit hacky, can we do this more robustly? XXX
                # v2: a regex instead of the fixed suffix list, which left a digit behind from iteration 11 on
                # ('_groupwise_iter1' also matched the start of '_groupwise_iter10')
                img_name = re.sub(r'_groupwise_iter\d+', '', img_name.replace('_ants-def0', ''))
                
                output_filename = f"{img_name}_groupwise_iter{iteration}"
                
                # Get the appropriate template for this slice (local or global)
                slice_template_fname = template_fnames[idx]
                
                future = executor.submit(
                    run_groupwise_registration_single,  # Use wrapper function
                    idx,
                    full_img_name,
                    slice_template_fname,  # Per-slice template (local or global)
                    iteration,
                    output_dir,
                    scaling_factor,
                    ignore_affine,
                    ignore_res,
                    output_filename,  # Pass the filename
                    cost_function
                )
                futures[future] = idx
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                idx = result['idx']
                results[idx] = result
                
                if result['success']:
                    # logging.info(f"✓ Slice {idx} registered")
                    logging.warning(f'Slice {idx} registration successful')
                else:
                    logging.warning(f"Slice {idx} failed: {result['error']}")
        
        # Check for failures
        failed = [r['idx'] for r in results if not r['success']]
        if failed:
            raise RuntimeError(f"Registration failed for slices: {failed}")
        
        # Determine if we should keep map/invmap files for this iteration
        # Keep if: using deformed source OR this is the last iteration
        using_deformed_source = (use_deformed_source_after_iteration is not None and 
                                  iteration >= use_deformed_source_after_iteration - 1) # We want to keep maps for the iteration just before we switch to deformed source, since those maps will be needed for the next iteration's registration
        is_last_iteration = (iteration == iterations - 1) # keep the maps for the last iteration, no matter what
        keep_maps = using_deformed_source or is_last_iteration
        
        if not keep_maps:
            # Delete map and invmap files for this iteration (intermediate original-source iterations)
            logging.warning(f"  Cleaning up intermediate map/invmap files for iteration {iteration}...")
            for result in results:
                if result['success'] and result['reg'] is not None:
                    # Get the base path from transformed_source and derive map/invmap paths
                    transformed_path = result['reg']['transformed_source']
                    base_path = transformed_path.replace('_ants-def0.nii.gz', '')
                    map_path = f"{base_path}_ants-map.nii.gz"
                    invmap_path = f"{base_path}_ants-invmap.nii.gz"
                    
                    for fpath in [map_path, invmap_path]:
                        if os.path.exists(fpath):
                            os.remove(fpath)
                            logging.info(f"    Removed: {os.path.basename(fpath)}")
        else:
            logging.warning(f"  Retaining map/invmap files for iteration {iteration} (deformed_source={using_deformed_source}, last_iter={is_last_iteration})")
        
        # Load newly registered images IN ORDER
        logging.warning("Loading registered images for next iteration...")
        registered_images = []
        new_images_fnames = []
        
        for result in results:
            reg_img = nib.load(result['reg']['transformed_source'])
            registered_images.append(reg_img)
            new_images_fnames.append(result['reg']['transformed_source'])
        
        # CRITICAL: Update images and filenames for next iteration
        images = registered_images
        images_fnames = new_images_fnames
        
        logging.warning(f"Completed groupwise iteration {iteration+1}/{iterations}")
        logging.warning(f"  Mean template variation: {mean_data.std():.3f}")
        
        # Diagnostic: compute template difference from previous iteration
        if iteration > 0:
            prev_template_fname = os.path.join(output_dir, f"groupwise_iter{iteration-1}_template.nii.gz")
            if os.path.exists(prev_template_fname):
                prev_template = nib.load(prev_template_fname).get_fdata()
                template_diff = np.abs(mean_data - prev_template).mean()
                template_max_diff = np.abs(mean_data - prev_template).max()
                logging.warning(f"  Template change from iter {iteration-1}: mean={template_diff:.4f}, max={template_max_diff:.4f}")
    
    logging.warning("=" * 80)
    logging.warning("Groupwise optimization complete")
    logging.warning(f"Saved {len(all_image_fnames)} deformation maps")
    logging.warning("=" * 80)
    
    return images

