"""Anchor the slice stack to a 3D MRI template (D99 cerebellum + brainstem) and produce per-slice MRI targets.

Frames (all images are handled as index-based ANTs images with identity direction and zero origin):
    - MRI frame:   (X, Y, Z) * mri_spacing_mm          (X = left-right, sections are sagittal)
    - stack frame: (i, j, s) * (res_mm, res_mm, slice_mm)  (s = slice index)
ANTs transforms map fixed (MRI) points to moving (stack) points, so `stack_to_mri_affine` returns a
transform that maps MRI -> stack coordinates. Resampling the MRI into the stack grid uses its inverse.

Workflow:
    1. stack_area_profile_fit    -> position of slice 0 relative to the MRI midline and the slice direction
    2. orientation_search_2d     -> in-plane flip / rotation / scale (one section vs. the predicted MRI plane)
    3. initial_affine_3d         -> 3D affine built from 1 and 2
    4. stack_to_mri_affine       -> refined 3D affine (MI, masked to our side)
    5. write_mri_slice_targets   -> one 2D MRI target per slice, in the slice file format of the pipeline

Only the affine is used for the MRI targets: a deformable MRI->stack warp would follow the stack's own drift.
"""
import os
import logging
import shutil
import numpy as np
import nibabel
import ants
from scipy import ndimage as ndi

from slice_registration_functions import create_affine


def load_mri(template_fname, mask_fname):
    """Load MRI template and mask as index-based arrays; returns (template, mask, spacing_mm, world_origin_x_mm)."""
    t = nibabel.load(template_fname)
    m = nibabel.load(mask_fname)
    if t.shape != m.shape:
        raise ValueError(f'template and mask shapes differ: {t.shape} vs {m.shape}')
    spacing = float(t.header.get_zooms()[0])
    return (np.asarray(t.dataobj, np.float32), np.asarray(m.dataobj) > 0, spacing, float(t.affine[0, 3]))


def stack_to_ants(stack, res_mm, slice_mm):
    return ants.from_numpy(np.ascontiguousarray(stack, dtype=np.float32), spacing=(res_mm, res_mm, slice_mm))


def downsample_stack(stack, res_mm, slice_mm, target_mm):
    """Smooth and resample the stack to isotropic target_mm, keeping the stack physical frame."""
    img = stack_to_ants(stack, res_mm, slice_mm)
    sig = [max(0.0, (target_mm ** 2 - s ** 2)) ** 0.5 / 2.355 for s in (res_mm, res_mm, slice_mm)]
    img = ants.smooth_image(img, sig, sigma_in_physical_coordinates=True)
    return ants.resample_image(img, (target_mm, target_mm, target_mm), use_voxels=False, interp_type=0)


def tissue_area_profile(stack, res_mm, threshold):
    return (stack > threshold).sum(axis=(0, 1)) * res_mm ** 2


def stack_area_profile_fit(stack_area_mm2, slice_mm, mri_mask, mri_spacing, mri_origin_x,
                           x0_range=(-12, 12), area_scales=np.arange(0.4, 1.41, 0.02)):
    """Fit the stack tissue-area curve to the MRI sagittal-area curve (slice spacing taken as nominal).
    Returns dict(x0_mm, direction, area_scale, rmse); x0_mm is the MRI world x of slice 0.
    The MRI curve is close to symmetric, so direction is only a weak estimate."""
    mri_area = mri_mask.sum(axis=(1, 2)) * mri_spacing ** 2
    xw = mri_origin_x + np.arange(mri_mask.shape[0]) * mri_spacing
    s = np.arange(len(stack_area_mm2))
    best = None
    for direction in (1, -1):
        for x0 in np.arange(x0_range[0], x0_range[1] + 1e-6, mri_spacing):
            a = np.interp(x0 + direction * s * slice_mm, xw, mri_area, left=0, right=0)
            for k in area_scales:
                e = np.mean((stack_area_mm2 - k * a) ** 2)
                if best is None or e < best['rmse'] ** 2:
                    best = dict(x0_mm=float(x0), direction=direction, area_scale=float(k), rmse=float(np.sqrt(e)))
    return best


def orientation_search_2d(stack_section, section_res_mm, mri_plane, mri_spacing, out_prefix,
                          angles=range(0, 360, 15)):
    """Similarity registration of one stack section (moving) to an MRI plane (fixed) from many start angles,
    with and without a flip of the first stack axis. Returns dict(flip, matrix, translation, center, mi)."""
    fi = ants.from_numpy(np.ascontiguousarray(mri_plane, dtype=np.float32), spacing=(mri_spacing, mri_spacing))
    best = None
    for flip in (False, True):
        sec = stack_section[::-1] if flip else stack_section
        mv = ants.from_numpy(np.ascontiguousarray(sec, dtype=np.float32), spacing=(section_res_mm, section_res_mm))
        c = np.array(ants.get_center_of_mass(fi))
        cm = np.array(ants.get_center_of_mass(mv))
        for ang in angles:
            tx = ants.create_ants_transform(transform_type='Euler2DTransform', dimension=2)
            tx.set_fixed_parameters(list(c))
            tx.set_parameters([np.deg2rad(ang), *(cm - c)])
            init_f = f'{out_prefix}orient_init.mat'
            ants.write_transform(tx, init_f)
            r = ants.registration(fi, mv, 'Similarity', initial_transform=init_f, aff_metric='mattes',
                                  aff_iterations=(200, 100, 50), aff_shrink_factors=(4, 2, 1),
                                  aff_smoothing_sigmas=(2, 1, 0), outprefix=f'{out_prefix}orient_')
            mi = ants.image_similarity(fi, r['warpedmovout'], metric_type='MattesMutualInformation')
            if best is None or mi < best['mi']:
                t = ants.read_transform(r['fwdtransforms'][0])
                p = np.array(t.parameters)
                best = dict(flip=flip, matrix=p[:4].reshape(2, 2), translation=p[4:6],
                            center=np.array(t.fixed_parameters), mi=float(mi), angle0=ang,
                            n_i=stack_section.shape[0])
    return best


def initial_affine_3d(orient, section_res_mm, ref_slice, ref_mri_x_index, mri_spacing, slice_mm, direction):
    """Build the 3D MRI->stack affine from the 2D orientation result.
    MRI (X, Y, Z) mm -> stack (i, j, s) mm:
        (u, v) = A [(Y, Z) - c] + t + c   (2D similarity, u along the possibly flipped stack axis i)
        i = (n_i - 1) * res - u  if flipped, else u ;  j = v
        s = direction * (X - ref_mri_x_index * mri_spacing) + ref_slice * slice_mm
    """
    A, t, c = orient['matrix'], orient['translation'], orient['center']
    b2 = -A @ c + t + c
    M = np.zeros((3, 3))
    b = np.zeros(3)
    if orient['flip']:
        M[0, 1:] = -A[0]
        b[0] = (orient['n_i'] - 1) * section_res_mm - b2[0]
    else:
        M[0, 1:] = A[0]
        b[0] = b2[0]
    M[1, 1:] = A[1]
    b[1] = b2[1]
    M[2, 0] = direction
    b[2] = -direction * ref_mri_x_index * mri_spacing + ref_slice * slice_mm
    tx = ants.create_ants_transform(transform_type='AffineTransform', dimension=3,
                                    matrix=M, translation=b, center=(0.0, 0.0, 0.0))
    return tx


def remove_shear(matrix, n_iter=100):
    """Nearest (Frobenius) matrix of the form D @ R to an MRI->stack matrix, with D diagonal (scales along the
    stack axes: in-plane shrinkage and section spacing) and R orthogonal (cutting angle and in-plane orientation;
    a reflection is allowed for the in-plane flip). A shear between the slice axis and the in-plane axes is what a
    linear slice-to-slice drift looks like, so it is removed here instead of being copied into the MRI targets."""
    D = np.ones(3)
    for _ in range(n_iter):
        U, _, Vt = np.linalg.svd(D[:, None] * matrix)
        R = U @ Vt
        D = np.einsum('ij,ij->i', matrix, R)
    return D[:, None] * R, D, R


def _affine_3d(F, Mv, init_f, Mmask, direction, x0, mri_sp, mri_ox, side_margin_mm, mask_dilation_mm,
               tissue_threshold, out_prefix, mri_center_mm):
    """3D affine (fixed = MRI masked to our side + margin, moving = downsampled stack), then shear removal."""
    xw = mri_ox + np.arange(Mmask.shape[0]) * mri_sp
    side = direction * (xw - x0) >= -side_margin_mm  # the stack runs from x0 in `direction`
    fmask = ndi.binary_dilation(Mmask & side[:, None, None], iterations=max(1, int(round(mask_dilation_mm / mri_sp))))
    Fm = ants.from_numpy(fmask.astype(np.float32), spacing=(mri_sp,) * 3)
    r = ants.registration(F, Mv, 'Affine', initial_transform=init_f, mask=Fm, aff_metric='mattes',
                          aff_iterations=(500, 250, 100, 50), aff_shrink_factors=(6, 4, 2, 1),
                          aff_smoothing_sigmas=(3, 2, 1, 0), outprefix=out_prefix + 'aff_')
    free_f = out_prefix + 'affine_free.mat'
    os.replace(r['fwdtransforms'][0], free_f)
    t = ants.read_transform(free_f)
    p = np.array(t.parameters)
    c = np.array(t.fixed_parameters)
    M, b = p[:9].reshape(3, 3), p[9:12]
    M2, D, R = remove_shear(M)
    # keep the mapping of the MRI tissue centre unchanged: M(x-c)+b+c == M2(x-c)+b2+c at x = mri_center_mm
    b2 = b + (M - M2) @ (mri_center_mm - c)
    t2 = ants.create_ants_transform(transform_type='AffineTransform', dimension=3,
                                    matrix=M2, translation=b2, center=tuple(c))
    aff_f = out_prefix + 'affine.mat'
    ants.write_transform(t2, aff_f)
    qc = dict(direction=direction, x0_mm=float(x0), scales_stack_axes=[float(v) for v in D],
              shear_removed_frob=float(np.linalg.norm(M - M2)),
              dice_init=_dice(ants.apply_transforms(F, Mv, [init_f]).numpy(), Mmask, fmask, tissue_threshold),
              dice_affine_free=_dice(r['warpedmovout'].numpy(), Mmask, fmask, tissue_threshold),
              dice_affine=_dice(ants.apply_transforms(F, Mv, [aff_f]).numpy(), Mmask, fmask, tissue_threshold))
    return aff_f, qc, Fm


def stack_to_mri_affine(stack, res_mm, slice_mm, mri_template_fname, mri_mask_fname, out_dir,
                        tissue_threshold=2.0, side_margin_mm=1.0, mask_dilation_mm=1.0,
                        work_mm=None, ref_slice=None, init_state=None, run_syn_qc=False):
    """Register the stack (i, j, s array) to the MRI.
    Returns (affine_file, qc). The affine maps MRI -> stack coordinates and has no shear (see remove_shear).
    init_state: qc dict of an earlier call (needs 'affine_file', 'direction', 'x0_mm'); if given, the area fit and
    the orientation search are skipped and the 3D affine starts from the earlier result."""
    os.makedirs(out_dir, exist_ok=True)
    pre = os.path.join(out_dir, 'mri_')
    T, Mmask, mri_sp, mri_ox = load_mri(mri_template_fname, mri_mask_fname)
    work_mm = mri_sp if work_mm is None else work_mm
    H = downsample_stack(stack, res_mm, slice_mm, work_mm).numpy()
    F = ants.from_numpy(T, spacing=(mri_sp,) * 3)
    Mv = ants.from_numpy(np.ascontiguousarray(H), spacing=(work_mm,) * 3)
    mri_center_mm = np.array(ndi.center_of_mass(Mmask)) * mri_sp
    common = dict(Mmask=Mmask, mri_sp=mri_sp, mri_ox=mri_ox, side_margin_mm=side_margin_mm,
                  mask_dilation_mm=mask_dilation_mm, tissue_threshold=tissue_threshold, mri_center_mm=mri_center_mm)

    if init_state is not None:
        aff_f, qc, Fm = _affine_3d(F, Mv, init_state['affine_file'], direction=init_state['direction'],
                                   x0=init_state['x0_mm'], out_prefix=pre, **common)
    else:
        # 1. area profile -> slice 0 position and direction
        fit = stack_area_profile_fit(tissue_area_profile(stack, res_mm, tissue_threshold), slice_mm, Mmask, mri_sp, mri_ox)
        logging.warning(f'MRI anchor: area profile fit {fit}')
        # 2. orientation on a mid-stack section
        #    (sections near the midline are dominated by the brainstem, which is only partly present in the stack)
        if ref_slice is None:
            ref_slice = stack.shape[2] // 2
        ref_h = min(int(round(ref_slice * slice_mm / work_mm)), H.shape[2] - 1)
        # the MRI is close to left-right symmetric, so both slice directions are run to the end
        # (orientation + 3D affine) and the one with the better final Dice is kept
        best = None
        for direction, x0 in [(fit['direction'], fit['x0_mm']), (-fit['direction'], -fit['x0_mm'])]:
            dpre = f'{pre}{"pos" if direction > 0 else "neg"}_'
            xi = int(round((x0 + direction * ref_slice * slice_mm - mri_ox) / mri_sp))
            orient = orientation_search_2d(H[..., ref_h], work_mm, T[xi], mri_sp, dpre)
            init_f = dpre + 'init_affine.mat'
            ants.write_transform(initial_affine_3d(orient, work_mm, ref_slice, xi, mri_sp, slice_mm, direction), init_f)
            aff_f, qc, Fm = _affine_3d(F, Mv, init_f, direction=direction, x0=x0, out_prefix=dpre, **common)
            qc.update(area_fit=fit, orientation_mi=orient['mi'], flip=orient['flip'])
            logging.warning(f'MRI anchor: direction {direction}: {qc}')
            if best is None or qc['dice_affine'] > best[1]['dice_affine']:
                best = (aff_f, qc, Fm)
        aff_f, qc, Fm = best
        shutil.copyfile(aff_f, pre + 'affine.mat')
        aff_f = pre + 'affine.mat'

    ants.image_write(ants.apply_transforms(F, Mv, [aff_f]), pre + 'stack_in_mri_affine.nii.gz')
    if run_syn_qc:
        rs = ants.registration(F, Mv, 'SyNOnly', initial_transform=aff_f, mask=Fm, syn_metric='mattes',
                               reg_iterations=(100, 50, 0), flow_sigma=4, total_sigma=2, outprefix=pre + 'synqc_')
        qc['dice_syn_qc'] = _dice(rs['warpedmovout'].numpy(), Mmask, Fm.numpy() > 0, tissue_threshold)
    qc['affine_file'] = aff_f
    logging.warning(f'MRI anchor: QC {qc}')
    return aff_f, qc


def _dice(warped_stack, mri_mask, region, threshold):
    hm = np.array([ndi.binary_fill_holes(p) for p in (warped_stack > threshold)])
    a, b = hm & region, mri_mask & region
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def write_mri_slice_targets(affine_f, stack_shape, res_mm, slice_mm, mri_template_fname, out_fnames, voxel_res):
    """Resample the MRI into the stack grid with the inverse affine and write one 2D target per slice.
    out_fnames[s] is written with the same affine convention as the pipeline slices (create_affine)."""
    T = nibabel.load(mri_template_fname)
    mri_sp = float(T.header.get_zooms()[0])
    F = ants.from_numpy(np.asarray(T.dataobj, np.float32), spacing=(mri_sp,) * 3)
    grid = ants.from_numpy(np.zeros(stack_shape, np.float32), spacing=(res_mm, res_mm, slice_mm))
    V = ants.apply_transforms(grid, F, [affine_f], whichtoinvert=[True], interpolator='linear').numpy()
    for s, fname in enumerate(out_fnames):
        sl = V[..., s]
        hdr = nibabel.Nifti1Header()
        hdr.set_data_shape(sl.shape)
        img = nibabel.Nifti1Image(sl, affine=create_affine(sl.shape, voxel_res=voxel_res), header=hdr)
        img.update_header()
        nibabel.save(img, fname)
    return V
