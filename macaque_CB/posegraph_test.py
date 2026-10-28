"""Offline test: pose-graph correction (slice_posegraph.py) against the nearest-neighbour jitter correction.

Both corrections are computed on the same raw stack (coreg0nl of a run made without a correction) and applied
offline in the same way (one more resampling each), so blur affects both equally. Measures:
  - block steps: rigid fit of the mean of 4 slices before each boundary to the mean of the 4 after it; the RMS
    displacement of the tissue (mm) and the rotation. A block of sections mounted at a common angle shows as a
    pair of large steps of opposite sign;
  - adjacent continuity: blur-equalized NCC and tissue-mask Dice (median, 5th percentile);
  - slow path change against the uncorrected stack (both corrections should keep it).

usage: uv run python macaque_CB/posegraph_test.py <run_dir> <rescale> <out_dir> [--workers 6]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel
from scipy import ndimage as ndi

import env_compat  # noqa: F401
import ants
import mri_anchor
from evaluate_registration import blur_equalized_ncc, adjacent_mask_dice, slow_path
from slice_jitter import jitter_poses, apply_poses
from slice_posegraph import posegraph_poses

MISSING = [32, 59, 120, 160, 189, 228]
WATCH = [33, 34, 35, 234, 235, 276, 277, 282, 283, 284, 324, 326]


def _block_fit(job):
    a, b, thr, res = job
    fi = ants.from_numpy(np.ascontiguousarray(b), spacing=(res, res))
    mv = ants.from_numpy(np.ascontiguousarray(a), spacing=(res, res))
    r = ants.registration(fi, mv, 'Rigid', aff_metric='GC', aff_random_sampling_rate=1.0, aff_iterations=(200, 100, 50),
                          aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0))
    t = ants.read_transform(r['fwdtransforms'][0])
    p = np.array(t.parameters); fc = np.array(t.fixed_parameters)
    A, tr = p[:4].reshape(2, 2), p[4:6]
    for f in set(r['fwdtransforms'] + r['invtransforms']):
        if os.path.isfile(f) and os.path.dirname(f) == '/tmp':
            os.remove(f)
    m = ndi.binary_fill_holes(b > thr)
    ii, jj = np.nonzero(m)
    pts = np.c_[ii, jj] * res
    disp = np.linalg.norm((pts - fc) @ A.T + tr + fc - pts, axis=1)
    return float(np.sqrt((disp ** 2).mean())), float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))


def block_steps(S, thr, res, k=4, workers=6):
    jobs = [(S[..., s - k + 1:s + 1].mean(2), S[..., s + 1:s + k + 1].mean(2), thr, res) for s in range(k - 1, S.shape[2] - k)]
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_block_fit, jobs, chunksize=4))
    rms = np.full(S.shape[2], np.nan); rot = np.full(S.shape[2], np.nan)
    for s, (a, b) in zip(range(k - 1, S.shape[2] - k), out):
        rms[s], rot[s] = a, b
    return rms, rot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir'); p.add_argument('rescale', type=int); p.add_argument('out')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--level', default='coreg0nl')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = a.rescale / 100.0
    thr = mri_anchor.tissue_threshold_for(res)
    img = nibabel.load(os.path.join(a.run_dir, f'zefir_{a.level}_stack.nii.gz'))
    S = np.asarray(img.dataobj, dtype=np.float32)

    stacks = {'none': S}
    pj, cj = jitter_poses(S, res, thr, exclude=MISSING, max_workers=a.workers)
    stacks['jitter'] = apply_poses(S, pj, cj, res)
    pg, cg = posegraph_poses(S, res, thr, exclude=MISSING, max_workers=a.workers)
    stacks['posegraph'] = apply_poses(S, pg, cg, res)
    np.save(os.path.join(a.out, 'poses_jitter.npy'), pj); np.save(os.path.join(a.out, 'poses_posegraph.npy'), pg)
    np.save(os.path.join(a.out, 'centroids.npy'), cg)

    summary, curves = {}, {}
    ref_path = slow_path(S, thr, res)
    for name, X in stacks.items():
        if name != 'none':
            nibabel.save(nibabel.Nifti1Image(X, img.affine), os.path.join(a.out, f'stack_{name}.nii.gz'))
        rms, rot = block_steps(X, thr, res, workers=a.workers)
        b = blur_equalized_ncc(X, thr); dc = adjacent_mask_dice(X, thr)
        dp = slow_path(X, thr, res) - ref_path
        dp -= np.nanmean(dp, axis=0)
        curves[name] = dict(rms=rms, rot=rot)
        summary[name] = dict(
            block_rms_median=float(np.nanmedian(rms)), block_rms_p95=float(np.nanpercentile(rms, 95)),
            block_rms_max=float(np.nanmax(rms)), n_boundaries_over_0p5mm=int(np.nansum(rms > 0.5)),
            blur_eq_ncc_median=float(np.median(b)), blur_eq_ncc_p5=float(np.percentile(b, 5)),
            mask_dice_median=float(np.median(dc)), mask_dice_p5=float(np.percentile(dc, 5)),
            slow_path_change_centroid_mm_max=float(np.abs(dp[:, :2]).max()),
            watch_rotation_deg={int(s): round(float(rot[s]), 1) for s in WATCH})
        print(name, json.dumps(summary[name]), flush=True)
    summary['poses'] = {n: dict(rotation_sd_deg=float(P[:, 0].std()), shift_sd_mm=float(np.linalg.norm(P[:, 1:], axis=1).std()))
                        for n, P in (('jitter', pj), ('posegraph', pg))}
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    for name, col in zip(curves, ('#8a8984', '#2a78d6', '#eb6834')):
        ax[0].plot(curves[name]['rms'], color=col, lw=1, label=name)
        ax[1].plot(curves[name]['rot'], color=col, lw=1, label=name)
    ax[2].plot(pj[:, 0], color='#2a78d6', lw=1, label='jitter rotation'); ax[2].plot(pg[:, 0], color='#eb6834', lw=1, label='pose-graph rotation')
    ax[0].set_ylabel('block step RMS (mm)'); ax[1].set_ylabel('block step rotation (deg)'); ax[2].set_ylabel('applied rotation (deg)')
    for x in ax:
        x.legend(loc='upper left', fontsize=8)
        for s in MISSING:
            x.axvline(s, color='#8a8984', lw=0.5, ls=':')
    ax[2].set_xlabel('slice / boundary')
    plt.tight_layout(); plt.savefig(os.path.join(a.out, 'block_steps.png'), dpi=60)


if __name__ == '__main__':
    main()
