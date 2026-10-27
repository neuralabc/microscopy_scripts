"""Offline test of an MRI-free drift control on the per-slice rotations.

The raw slices (coreg0nl) are the separately scanned sections placed by their image centres (no registration),
so their rotations come from mounting each section by hand and should be independent from section to section.
The rotation each slice gets from raw to final (STAGE1 map) should then be random along the stack too; a slow
component is either algorithmic drift or a systematic mounting habit. (Raw translations carry no physical
position - each image is its own scan - so translation cannot be controlled this way.)

Per run: the rigid part (rotation, centroid shift) of each slice's STAGE1 map, fitted over tissue; its slow part
(Gaussian, sigma slices); how large it is against what independent mounting angles would leave; and how much of
it the runs share (shared = from the data, run-specific = from the algorithm). Then the control: rotate each
final slice of one run about its tissue centroid by minus its slow rotation, and compare adjacent-slice NCC.

usage: uv run python drift_control_test.py <out_dir> --rescale 40 name=<run_dir> [name=<run_dir> ...]
"""
import argparse
import glob
import json
import os

import numpy as np
import nibabel
from scipy import ndimage as ndi

import mri_anchor  # tissue_threshold_for (density-based threshold)
from evaluate_registration import adjacent_ncc


def slice_poses(run_dir, level, thr, subject='zefir'):
    """Per slice: (rotation_deg, dx, dy) of the raw->final rigid part, from the coordinate map
    (target voxel -> source voxel) fitted with an affine over the final slice's tissue; shift in voxels."""
    maps = sorted(glob.glob(os.path.join(run_dir, f'{subject}_[0-9][0-9][0-9][0-9]_*_{level}_ants-map.nii.gz')))
    out = np.full((len(maps), 3), np.nan)
    for s, f in enumerate(maps):
        m = nibabel.load(f).get_fdata()
        d0 = nibabel.load(f.replace('_ants-map.nii.gz', '_ants-def0.nii.gz')).get_fdata()
        tissue = ndi.binary_fill_holes(d0 > thr)
        ii, jj = np.nonzero(tissue)
        if len(ii) < 20:
            continue
        tgt = np.stack([ii, jj], axis=1).astype(float)
        coef, *_ = np.linalg.lstsq(np.c_[tgt, np.ones(len(ii))], m[ii, jj, :2], rcond=None)
        A, b = coef[:2].T, coef[2]                      # src = A tgt + b
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt                                      # rotation part of src <- tgt
        theta_src = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        c = tgt.mean(axis=0)
        out[s] = [-theta_src, *(c - (A @ c + b))]       # correction raw -> final
    return out


def low_pass(x, sigma):
    x = x.copy()
    good = ~np.isnan(x)
    x[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), x[good])
    return ndi.gaussian_filter1d(x, sigma, mode='nearest')


def rotate_slices(stack, angles_deg, thr):
    """Rotate each slice about its tissue centroid by the given angle (degrees), linear interpolation."""
    out = np.zeros_like(stack)
    for s in range(stack.shape[2]):
        img = stack[..., s]
        t = ndi.binary_fill_holes(img > thr)
        if t.sum() < 20 or abs(angles_deg[s]) < 1e-3:
            out[..., s] = img
            continue
        c = np.array(ndi.center_of_mass(t))
        a = np.radians(angles_deg[s])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        # output voxel p takes input at R^T (p - c) + c (rotation by +a about c)
        out[..., s] = ndi.affine_transform(img, R.T, offset=c - R.T @ c, order=1, mode='constant', cval=0.0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('runs', nargs='+', help='name=run_dir')
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--level', default='coreg12nl_win12_rigsyn_4')
    p.add_argument('--sigmas', default='10,15,20')
    p.add_argument('--control-run', default=None, help='run to apply the rotation control to (default: first)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_mm = a.rescale * 10 / 1000.0
    thr = mri_anchor.tissue_threshold_for(res_mm)
    sigmas = [float(v) for v in a.sigmas.split(',')]
    runs = dict(spec.split('=', 1) for spec in a.runs)

    poses = {name: slice_poses(d.rstrip('/') + '/', a.level, thr) for name, d in runs.items()}
    for name, pz in poses.items():
        pz[:, 1:] *= res_mm                              # shift in mm
        np.save(os.path.join(a.out, f'poses_{name}.npy'), pz)

    summary = {}
    for name, pz in poses.items():
        th = pz[:, 0]
        hf = th - low_pass(th, 3)                        # section-to-section part (mounting + noise)
        row = dict(rotation_sd_deg=float(np.nanstd(th)), rotation_hf_sd_deg=float(np.nanstd(hf)))
        for sg in sigmas:
            lf = low_pass(th, sg)
            lf = lf - lf.mean()
            # independent angles with sd = hf sd, smoothed by a Gaussian of sigma sg: sd / sqrt(2 sqrt(pi) sg)
            row[f'sigma{sg:g}_slow_sd_deg'] = float(lf.std())
            row[f'sigma{sg:g}_slow_sd_expected_if_random_deg'] = float(np.nanstd(hf) / np.sqrt(2 * np.sqrt(np.pi) * sg))
            row[f'sigma{sg:g}_slow_range_deg'] = float(lf.max() - lf.min())
        summary[name] = row
    names = list(poses)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for sg in sigmas:
                for k, lab in enumerate(['rotation', 'dx', 'dy']):
                    x = low_pass(poses[names[i]][:, k], sg); y = low_pass(poses[names[j]][:, k], sg)
                    x, y = x - x.mean(), y - y.mean()
                    summary[f'{names[i]} vs {names[j]} sigma{sg:g} {lab}'] = dict(
                        r=float(np.corrcoef(x, y)[0, 1]), rms_diff=float(np.sqrt(np.mean((x - y) ** 2))),
                        rms_a=float(x.std()), rms_b=float(y.std()))

    ctrl = a.control_run or names[0]
    stack = nibabel.load(os.path.join(runs[ctrl], f'zefir_{a.level}_stack.nii.gz')).get_fdata().astype(np.float32)
    ncc0 = adjacent_ncc(stack, thr)
    for sg in sigmas:
        lf = low_pass(poses[ctrl][:, 0], sg)
        corrected = rotate_slices(stack, -(lf - lf.mean()), thr)
        ncc1 = adjacent_ncc(corrected, thr)
        summary[f'control {ctrl} sigma{sg:g}'] = dict(
            ncc_median_before=float(np.median(ncc0)), ncc_median_after=float(np.median(ncc1)),
            ncc_p5_before=float(np.percentile(ncc0, 5)), ncc_p5_after=float(np.percentile(ncc1, 5)),
            max_abs_rotation_applied_deg=float(np.abs(lf - lf.mean()).max()))
        if sg == sigmas[len(sigmas) // 2]:
            nibabel.save(nibabel.Nifti1Image(corrected, nibabel.load(os.path.join(
                runs[ctrl], f'zefir_{a.level}_stack.nii.gz')).affine),
                os.path.join(a.out, f'{ctrl}_{a.level}_rotation_controlled_sigma{sg:g}_stack.nii.gz'))
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for k, v in summary.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for k, lab in enumerate(['rotation correction (deg)', 'centroid shift x (mm)', 'centroid shift y (mm)']):
        for name, pz in poses.items():
            line, = ax[k].plot(pz[:, k], '.', ms=2, alpha=0.35)
            ax[k].plot(low_pass(pz[:, k], 15), color=line.get_color(), label=f'{name} (sigma 15)')
        ax[k].set_ylabel(lab); ax[k].legend(ncol=len(poses), fontsize=8)
        for x in (113, 252):
            ax[k].axvline(x, color='r', ls=':')
    ax[2].set_xlabel('slice')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, 'drift_poses.png'), dpi=60)


if __name__ == '__main__':
    main()
