"""Compare slice-registration runs (output dirs of run_slice_registration_optimized_newReg_sdf_v2_local.py).

Measures per run:
  - continuity: NCC of adjacent slices (high spatial frequency along the stack)
  - MRI agreement: an independent shear-free affine fit of the final stack to the MRI template, then per slice
    the Dice and the tissue-centroid offset between the stack and its MRI plane (low frequency: drift, shape).
    Runs that used the MRI anchor were optimised toward this target, so for them it is not independent.
  - warp size: non-affine part of the final STAGE1 maps (coreg0nl -> final rigsyn level) inside tissue

usage: uv run python evaluate_registration.py --rescale 40 --out <dir> name=<run_dir> [name=<run_dir> ...]
"""
import argparse
import glob
import json
import os

import numpy as np
import nibabel
from scipy import ndimage as ndi

import env_compat  # noqa: F401  (ANTs shims, threads, OOM priority)
import mri_anchor

MRI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_extras', 'cerebellum_target')
MRI_T = os.path.join(MRI_DIR, 'D99_cerebellum_brainstem_template_crop.nii.gz')
MRI_M = os.path.join(MRI_DIR, 'D99_cerebellum_brainstem_mask_crop.nii.gz')


def adjacent_ncc(stack, threshold):
    out = []
    for i in range(stack.shape[2] - 1):
        a, b = stack[..., i], stack[..., i + 1]
        m = (a > threshold) | (b > threshold)
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        out.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)))
    return np.array(out)


def mri_agreement(stack, res_mm, slice_mm, out_dir, threshold, mri_threshold=5.0):
    aff_f, qc = mri_anchor.stack_to_mri_affine(stack, res_mm, slice_mm, MRI_T, MRI_M, out_dir)
    fnames = [os.path.join(out_dir, f'mri_plane_{s:04d}.nii.gz') for s in range(stack.shape[2])]
    V = mri_anchor.write_mri_slice_targets(aff_f, stack.shape, res_mm, slice_mm, MRI_T, fnames,
                                           [res_mm * 1000, res_mm * 1000, slice_mm * 1000])
    for f in fnames:
        os.remove(f)
    dice, offset = [], []
    for s in range(stack.shape[2]):
        h = ndi.binary_fill_holes(stack[..., s] > threshold)
        m = V[..., s] > mri_threshold
        if h.sum() == 0 or m.sum() == 0:
            dice.append(np.nan); offset.append(np.nan); continue
        dice.append(2 * (h & m).sum() / (h.sum() + m.sum()))
        offset.append(np.linalg.norm(np.array(ndi.center_of_mass(h)) - np.array(ndi.center_of_mass(m))) * res_mm)
    return np.array(dice), np.array(offset), qc


def warp_size(run_dir, level, res_mm, threshold):
    """RMS (mm) of the non-affine part of each slice map inside tissue, and RMS of the full displacement."""
    nonaff, total = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, f'*_{level}_ants-map.nii.gz'))):
        m = nibabel.load(f).get_fdata()
        d0 = f.replace('_ants-map.nii.gz', '_ants-def0.nii.gz')
        tissue = nibabel.load(d0).get_fdata() > threshold if os.path.isfile(d0) else np.ones(m.shape[:2], bool)
        ii, jj = np.nonzero(tissue)
        if len(ii) < 10:
            continue
        src = m[ii, jj, :]
        grid = np.stack([ii, jj, np.ones_like(ii)], axis=1).astype(float)
        coef, *_ = np.linalg.lstsq(grid, src, rcond=None)
        resid = src - grid @ coef
        nonaff.append(np.sqrt((resid ** 2).sum(axis=1).mean()) * res_mm)
        total.append(np.sqrt(((src - grid[:, :2]) ** 2).sum(axis=1).mean()) * res_mm)
    return np.array(nonaff), np.array(total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='name=run_dir')
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--subject', default='zefir')
    p.add_argument('--threshold', type=float, default=2.0)
    p.add_argument('--stage1_level', default='coreg12nl_win12_rigsyn_4')
    p.add_argument('--groupwise_iters', type=int, default=10)
    a = p.parse_args()
    res_mm, slice_mm = a.rescale * 10 / 1000.0, 0.05
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for spec in a.runs:
        name, run_dir = spec.split('=', 1)
        run_dir = run_dir.rstrip('/') + '/'
        levels = {'stage1': a.stage1_level,
                  'groupwise': f'{a.stage1_level}_groupwise_iter{a.groupwise_iters - 1}'}
        summary[name] = {}
        for lname, level in levels.items():
            sf = f'{run_dir}{a.subject}_{level}_stack.nii.gz'
            if not os.path.isfile(sf):
                print(f'{name}: missing {sf}')
                continue
            stack = nibabel.load(sf).get_fdata().astype(np.float32)
            ncc = adjacent_ncc(stack, a.threshold)
            dice, offset, qc = mri_agreement(stack, res_mm, slice_mm, os.path.join(a.out, name, lname), a.threshold)
            np.savez(os.path.join(a.out, name, lname, 'per_slice.npz'), ncc=ncc, dice=dice, offset_mm=offset)
            summary[name][lname] = dict(ncc_median=float(np.median(ncc)), ncc_p5=float(np.percentile(ncc, 5)),
                                        mri_dice_3d=qc['dice_affine'], mri_dice_slice_median=float(np.nanmedian(dice)),
                                        mri_offset_mm_median=float(np.nanmedian(offset)),
                                        mri_offset_mm_p95=float(np.nanpercentile(offset, 95)))
        nonaff, total = warp_size(run_dir, a.stage1_level, res_mm, a.threshold)
        summary[name]['stage1_warp'] = dict(nonaffine_rms_mm_median=float(np.median(nonaff)),
                                            total_rms_mm_median=float(np.median(total)), n=len(nonaff))
        print(name, json.dumps(summary[name], indent=1))
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
