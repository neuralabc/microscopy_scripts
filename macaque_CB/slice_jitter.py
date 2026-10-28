"""Rigid slice-to-slice jitter correction that keeps the slow path (no warping).

Each iteration fits every slice rigidly (global correlation) to the mean of its nearest valid neighbours, removes
the slow part of the poses along the stack (Gaussian, `sigma` slices) and adds a fraction (`step`) of the fast part
to each slice's pose. The slow path (where the stack sits as a whole, and how it bends) is kept, so this cannot
add drift; it only removes section-to-section jitter (mounting rotation, framing offsets).

`correct_level_in_place` folds the poses into an existing registration level (e.g. coreg0nl): its forward map,
inverse map and deformed slice are rewritten with the same names and formats, and the native slice is resampled
once through the composed map. Offline tests (2026-10-27, rescale 40, run A): from the raw coreg0nl stack, adjacent
NCC 0.41 -> 0.82 with rotation SD 5 deg and shift SD 0.2 mm; see REGISTRATION_NOTES_local.md.

Poses are (theta_deg, tx_mm, ty_mm): rotate about the slice's tissue centroid (voxel `centroids[s]`), then translate.
"""
import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel
from scipy import ndimage as ndi


def _rigid_to_target(args):
    """Pose that moves slice `mov` onto `target` (rigid, GC metric); None if either has too little tissue."""
    import ants
    mov, target, thr, res_mm = args
    tm = ndi.binary_fill_holes(mov > thr)
    if tm.sum() < 20 or (target > thr).sum() < 20:
        return None
    fi = ants.from_numpy(np.ascontiguousarray(target, np.float32), spacing=(res_mm, res_mm))
    mv = ants.from_numpy(np.ascontiguousarray(mov, np.float32), spacing=(res_mm, res_mm))
    r = ants.registration(fi, mv, 'Rigid', aff_metric='GC', aff_random_sampling_rate=1.0,
                          aff_iterations=(200, 100, 50), aff_shrink_factors=(4, 2, 1),
                          aff_smoothing_sigmas=(2, 1, 0), verbose=False)
    t = ants.read_transform(r['fwdtransforms'][0])  # target point -> slice point
    for f in set(r['fwdtransforms'] + r['invtransforms']):
        if os.path.isfile(f) and os.path.dirname(f) == '/tmp':
            os.remove(f)
    p = np.array(t.parameters)
    fc = np.array(t.fixed_parameters)
    A, b = p[:4].reshape(2, 2), p[4:6]
    c = np.array(ndi.center_of_mass(tm)) * res_mm
    c_new = np.linalg.inv(A) @ (c - b - fc) + fc     # where the slice centroid belongs on the target
    return [-np.degrees(np.arctan2(A[1, 0], A[0, 0])), *(c_new - c)]


def _neighbours(s, valid):
    """Nearest valid slice on each side."""
    out = []
    for direction in (-1, 1):
        k = s + direction
        while 0 <= k < len(valid) and not valid[k]:
            k += direction
        if 0 <= k < len(valid):
            out.append(k)
    return out


def _low_pass(x, sigma):
    x = x.copy()
    good = ~np.isnan(x)
    if not good.any():
        return np.zeros_like(x), np.zeros_like(x)
    x[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), x[good])
    return ndi.gaussian_filter1d(x, sigma, mode='nearest'), x


def pose_to_voxel(pose, centroid_vox, res_mm):
    """(R, c, t_vox) of the forward mapping x_new = R (x - c) + c + t in voxel units."""
    a = np.radians(pose[0])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return R, np.asarray(centroid_vox, float), np.array(pose[1:3]) / res_mm


def apply_poses(stack, poses, centroids, res_mm):
    out = np.zeros_like(stack)
    for s in range(stack.shape[2]):
        R, c, t = pose_to_voxel(poses[s], centroids[s], res_mm)
        out[..., s] = ndi.affine_transform(stack[..., s], R.T, offset=c - R.T @ (c + t), order=1, cval=0.0)
    return out


def adjacent_ncc(stack, thr):
    out = []
    for s in range(stack.shape[2] - 1):
        a, b = stack[..., s], stack[..., s + 1]
        m = (a > thr) | (b > thr)
        x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
        out.append((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))
    return np.array(out)


def jitter_poses(stack, res_mm, thr, sigma=15.0, iters=6, step=0.5, exclude=(), max_workers=1):
    """Poses (n x 3) and centroids (n x 2, voxels) of the jitter correction for a stack (x, y, slice)."""
    n = stack.shape[2]
    valid = np.ones(n, bool)
    valid[[s for s in exclude if 0 <= s < n]] = False
    centroids = np.array([ndi.center_of_mass(ndi.binary_fill_holes(stack[..., s] > thr))
                          if (stack[..., s] > thr).sum() > 20 else (np.array(stack.shape[:2]) - 1) / 2
                          for s in range(n)])
    total = np.zeros((n, 3))
    cur = stack
    logging.warning(f'Jitter correction: adjacent NCC median {np.median(adjacent_ncc(cur, thr)):.3f} before')
    for it in range(iters):
        jobs = []
        for s in range(n):
            nb = _neighbours(s, valid)
            target = np.mean([cur[..., k] for k in nb], axis=0) if nb else None
            jobs.append((cur[..., s], target, thr, res_mm) if (valid[s] and nb) else None)
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                res = list(ex.map(_rigid_or_none, jobs))
        else:
            res = [_rigid_or_none(j) for j in jobs]
        raw = np.array([r if r is not None else [np.nan] * 3 for r in res], float)
        for k in range(3):
            slow, filled = _low_pass(raw[:, k], sigma)
            fast = filled - slow
            fast[~valid] = 0.0
            total[:, k] += step * fast
            total[:, k] -= _low_pass(total[:, k], sigma)[0]   # keep the slow path exactly
        total[~valid] = 0.0
        cur = apply_poses(stack, total, centroids, res_mm)
        logging.warning(f'Jitter correction iteration {it + 1}/{iters}: adjacent NCC median '
                        f'{np.median(adjacent_ncc(cur, thr)):.3f}, rotation SD {total[:, 0].std():.2f} deg, '
                        f'shift SD {np.linalg.norm(total[:, 1:], axis=1).std():.3f} mm')
    return total, centroids


def _rigid_or_none(job):
    return None if job is None else _rigid_to_target(job)


def correct_level_in_place(output_dir, subject, all_image_fnames, level, res_mm, voxel_res, thr, sigma=15.0,
                           iters=6, step=0.5, exclude=(), max_workers=1, zfill_num=4, poses_fn=None):
    """Fold the jitter correction into `level`: rewrite each slice's forward map, inverse map and deformed slice
    (root copy and the copy beside the maps), same names and formats. Writes `{subject}_{level}_jitter_poses.csv`.
    poses_fn: the function that computes the poses (default jitter_poses; e.g. slice_posegraph.posegraph_poses),
    same signature and pose convention. Returns the poses."""
    import pandas as pd
    from slice_registration_functions import create_affine
    names = [os.path.basename(f).split('.')[0] for f in all_image_fnames]
    pre = [f'{subject}_{str(i).zfill(zfill_num)}_{nm}' for i, nm in enumerate(names)]
    def0 = [os.path.join(output_dir, f'{p}_{level}_ants-def0.nii.gz') for p in pre]
    stack = np.stack([nibabel.load(f).get_fdata().astype(np.float32) for f in def0], axis=2)
    poses, centroids = (poses_fn or jitter_poses)(stack, res_mm, thr, sigma=sigma, iters=iters, step=step,
                                                  exclude=exclude, max_workers=max_workers)
    for s, p in enumerate(pre):
        if np.allclose(poses[s], 0):
            continue
        maps = glob.glob(os.path.join(output_dir, '**', f'{p}_{level}_ants-map.nii.gz'), recursive=True)
        invs = glob.glob(os.path.join(output_dir, '**', f'{p}_{level}_ants-invmap.nii.gz'), recursive=True)
        if len(maps) != 1 or len(invs) != 1:
            raise FileNotFoundError(f'{p}: expected one {level} map and inverse map, found {maps} {invs}')
        R, c, t = pose_to_voxel(poses[s], centroids[s], res_mm)
        mimg = nibabel.load(maps[0]); m = mimg.get_fdata()
        # forward map: new canvas point p takes the old map at R^T (p - c - t) + c
        ii, jj = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing='ij')
        q = np.einsum('ab,bij->aij', R.T, np.stack([ii - c[0] - t[0], jj - c[1] - t[1]])) + c[:, None, None]
        # inside only where all map entries used by the linear interpolation are valid: at the border of the valid
        # region, interpolating with the zeros outside gives coordinates inside the slice and copies tissue to the
        # edges (seen as dotted streaks in run JR, 2026-10-27)
        inside = ndi.map_coordinates(np.any(m != 0, axis=2).astype(np.float32), q, order=1, cval=0.0) > 0.999
        m_new = np.stack([ndi.map_coordinates(m[..., k], q, order=1, cval=0.0) for k in range(m.shape[2])], axis=2)
        m_new[~inside] = 0.0
        nibabel.save(nibabel.Nifti1Image(m_new, mimg.affine, mimg.header), maps[0])
        # inverse map: native point -> old canvas point -> forward pose
        iimg = nibabel.load(invs[0]); iv = iimg.get_fdata()
        valid_iv = np.any(iv != 0, axis=2)
        x = iv[..., :2] - c
        iv_new = iv.copy()
        iv_new[..., :2] = np.einsum('ab,ijb->ija', R, x) + c + t
        iv_new[~valid_iv] = 0.0
        nibabel.save(nibabel.Nifti1Image(iv_new, iimg.affine, iimg.header), invs[0])
        # deformed slice: native slice through the composed map, one interpolation
        native = nibabel.load(os.path.join(output_dir, f'{p}.nii.gz')).get_fdata().astype(np.float32)
        d = ndi.map_coordinates(native, [m_new[..., 0], m_new[..., 1]], order=1, cval=0.0)
        d[~inside] = 0.0
        hdr = nibabel.Nifti1Header(); hdr.set_data_shape(d.shape)
        out = nibabel.Nifti1Image(d.astype(np.float32), affine=create_affine(d.shape, voxel_res=voxel_res), header=hdr)
        for f in [def0[s]] + glob.glob(os.path.join(output_dir, '**', os.path.basename(def0[s])), recursive=True):
            nibabel.save(out, f)
    pd.DataFrame({'slice_idx': range(len(pre)), 'theta_deg': poses[:, 0], 'tx_mm': poses[:, 1], 'ty_mm': poses[:, 2],
                  'centroid_i_vox': centroids[:, 0], 'centroid_j_vox': centroids[:, 1]}).to_csv(
        os.path.join(output_dir, f'{subject}_{level}_jitter_poses.csv'), index=False)
    return poses
