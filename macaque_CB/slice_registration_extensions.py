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
