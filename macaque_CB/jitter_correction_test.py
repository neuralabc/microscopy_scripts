"""Offline test: rigid slice-to-slice jitter correction that keeps the slow path (no warping).

Each iteration: rigid fit (global correlation) of every slice to the mean of its two neighbours, the slow part of
the resulting poses (Gaussian, sigma slices) is removed, and only the fast part is added to each slice's pose.
Poses are kept per slice and every iteration resamples the input stack once (no accumulated blur).
Rotation is about each slice's tissue centroid; translation is in mm.

Measures per iteration: adjacent NCC (median, 5th pct), adjacent tissue-mask Dice (median, 5th pct), size of the
applied correction, and the slow path of the applied poses (should stay ~0).

usage: uv run python jitter_correction_test.py <stack.nii.gz> <in-plane res mm> <out_dir> [--sigma 15] [--iters 3]
       [--block 1]  (in-plane block SUM before fitting, to speed up fine stacks)
"""
import argparse
import json
import os

import numpy as np
import nibabel
import ants
from scipy import ndimage as ndi

import env_compat  # noqa: F401
import mri_anchor


def adjacent_measures(stack, thr):
    ncc, dice = [], []
    masks = [ndi.binary_fill_holes(stack[..., s] > thr) for s in range(stack.shape[2])]
    for s in range(stack.shape[2] - 1):
        a, b = stack[..., s], stack[..., s + 1]
        m = masks[s] | masks[s + 1]
        x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
        ncc.append((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))
        dice.append(2 * (masks[s] & masks[s + 1]).sum() / (masks[s].sum() + masks[s + 1].sum() + 1e-12))
    return np.array(ncc), np.array(dice)


def apply_poses(stack, poses, centroids, res_mm):
    """poses[s] = (theta_deg, tx_mm, ty_mm): rotate about the slice's centroid, then translate."""
    out = np.zeros_like(stack)
    for s in range(stack.shape[2]):
        th, tx, ty = poses[s]
        if abs(th) < 1e-6 and abs(tx) < 1e-9 and abs(ty) < 1e-9:
            out[..., s] = stack[..., s]
            continue
        a = np.radians(th)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = centroids[s]
        t = np.array([tx, ty]) / res_mm
        # output p takes input at R^T (p - c - t) + c
        out[..., s] = ndi.affine_transform(stack[..., s], R.T, offset=c - R.T @ (c + t), order=1, cval=0.0)
    return out


def fit_to_neighbours(stack, thr, res_mm):
    """Rigid fit of each slice (moving) to the mean of its neighbours (fixed); returns the pose that moves the
    slice onto the target: (theta_deg about the slice centroid, tx_mm, ty_mm)."""
    n = stack.shape[2]
    poses = np.full((n, 3), np.nan)
    for s in range(n):
        nb = [k for k in (s - 1, s + 1) if 0 <= k < n]
        target = np.mean([stack[..., k] for k in nb], axis=0)
        mov = stack[..., s]
        tm = ndi.binary_fill_holes(mov > thr)
        if tm.sum() < 20 or (target > thr).sum() < 20:
            continue
        fi = ants.from_numpy(np.ascontiguousarray(target, np.float32), spacing=(res_mm, res_mm))
        mv = ants.from_numpy(np.ascontiguousarray(mov, np.float32), spacing=(res_mm, res_mm))
        r = ants.registration(fi, mv, 'Rigid', aff_metric='GC', aff_random_sampling_rate=1.0,
                              aff_iterations=(200, 100, 50), aff_shrink_factors=(4, 2, 1),
                              aff_smoothing_sigmas=(2, 1, 0), verbose=False)
        t = ants.read_transform(r['fwdtransforms'][0])  # fixed (target) point -> moving (slice) point
        p = np.array(t.parameters); fc = np.array(t.fixed_parameters)
        A = p[:4].reshape(2, 2); b = p[4:6]
        c = np.array(ndi.center_of_mass(tm)) * res_mm   # slice centroid (mm)
        # inverse maps the slice onto the target: q = A^-1 (x - b - fc) + fc; rotation by -angle(A)
        Ainv = np.linalg.inv(A)
        c_new = Ainv @ (c - b - fc) + fc
        poses[s] = [-np.degrees(np.arctan2(A[1, 0], A[0, 0])), *(c_new - c)]
        for f in r['fwdtransforms'] + r['invtransforms']:
            if os.path.isfile(f) and f.startswith('/tmp'):
                os.remove(f)
    return poses


def low_pass(x, sigma):
    x = x.copy()
    good = ~np.isnan(x)
    x[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), x[good])
    return ndi.gaussian_filter1d(x, sigma, mode='nearest'), x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stack'); p.add_argument('res_mm', type=float); p.add_argument('out')
    p.add_argument('--sigma', type=float, default=15.0)
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--block', type=int, default=1)
    p.add_argument('--step', type=float, default=1.0, help='fraction of the fast correction applied per iteration')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img = nibabel.load(a.stack)
    S = img.get_fdata().astype(np.float32)
    res = a.res_mm
    if a.block > 1:
        f = a.block
        h, w = S.shape[0] // f * f, S.shape[1] // f * f
        S = S[:h, :w].reshape(h // f, f, w // f, f, S.shape[2]).sum(axis=(1, 3))
        res *= f
    thr = mri_anchor.tissue_threshold_for(res)
    n = S.shape[2]
    centroids = np.array([ndi.center_of_mass(ndi.binary_fill_holes(S[..., s] > thr)) if (S[..., s] > thr).sum() > 20
                          else (np.array(S.shape[:2]) - 1) / 2 for s in range(n)])
    total = np.zeros((n, 3))
    hist = []
    ncc, dice = adjacent_measures(S, thr)
    hist.append(dict(iter=0, ncc_median=float(np.median(ncc)), ncc_p5=float(np.percentile(ncc, 5)),
                     dice_median=float(np.median(dice)), dice_p5=float(np.percentile(dice, 5))))
    print(hist[-1], flush=True)
    cur = S
    for it in range(1, a.iters + 1):
        raw = fit_to_neighbours(cur, thr, res)
        fast = np.zeros_like(raw)
        for k in range(3):
            slow, filled = low_pass(raw[:, k], a.sigma)
            fast[:, k] = filled - slow
        total += a.step * fast
        # keep the slow path of the total exactly where it was
        for k in range(3):
            total[:, k] -= low_pass(total[:, k], a.sigma)[0]
        cur = apply_poses(S, total, centroids, res)
        ncc, dice = adjacent_measures(cur, thr)
        hist.append(dict(iter=it, ncc_median=float(np.median(ncc)), ncc_p5=float(np.percentile(ncc, 5)),
                         dice_median=float(np.median(dice)), dice_p5=float(np.percentile(dice, 5)),
                         applied_rot_sd_deg=float(total[:, 0].std()),
                         applied_shift_sd_mm=float(np.linalg.norm(total[:, 1:], axis=1).std()),
                         applied_shift_max_mm=float(np.linalg.norm(total[:, 1:], axis=1).max()),
                         applied_slow_max=float(np.abs(np.array([low_pass(total[:, k], a.sigma)[0]
                                                                 for k in range(3)])).max())))
        print(hist[-1], flush=True)
    np.save(os.path.join(a.out, 'poses.npy'), total)
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(dict(stack=a.stack, res_mm=res, sigma=a.sigma, history=hist), f, indent=1)
    aff = img.affine.copy()
    if a.block > 1:
        aff[:3, :2] *= a.block
    nibabel.save(nibabel.Nifti1Image(cur, aff), os.path.join(a.out, 'jitter_corrected_stack.nii.gz'))


if __name__ == '__main__':
    main()
