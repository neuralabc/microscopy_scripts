"""Offline test: is there a real low-frequency signal in per-slice rigid poses to the MRI?

usage: uv run python mri_pose_consistency.py <out_dir> [stack.nii.gz] [in-plane res mm] [in-plane block-sum factor]
(run 2026-10-27 on run A rescale 40 and on the rescale-5 groupwise stack: see REGISTRATION_NOTES_local.md 4.4)

For run A's stack: global shear-free fits to D99 (two reference slices), MRI planes in the stack grid, per-slice
2D rigid fit (mask-based, mean squares) of each histology slice to its MRI plane, then smoothing along the stack.
Signal tests: split-half (even vs odd slices) and cross-fit (fit 1 vs fit 2) agreement of the smoothed curves,
and the amplitude of the smooth part against the slice-to-slice residual.
"""
import sys, os, json
sys.path.insert(0, '/home/chris/Documents/code/microscopy_scripts/macaque_CB')
import env_compat  # noqa
import numpy as np, nibabel as nib, ants, logging
from scipy import ndimage as ndi
import mri_anchor as ma
from evaluate_registration import MRI_T, MRI_M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

logging.disable(logging.WARNING)
OUT = sys.argv[1]
STACK = sys.argv[2] if len(sys.argv) > 2 else '/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/zefir_sliceReg_optimized_v2_rescale_40_local/zefir_coreg12nl_win12_rigsyn_4_stack.nii.gz'
RES, SL = (float(sys.argv[3]) if len(sys.argv) > 3 else 0.4), 0.05
os.makedirs(OUT, exist_ok=True)
S = nib.load(STACK).get_fdata().astype(np.float32)
F_DS = int(sys.argv[4]) if len(sys.argv) > 4 else 1  # in-plane block SUM (counts per voxel stay consistent)
if F_DS > 1:
    a, b = S.shape[0] // F_DS * F_DS, S.shape[1] // F_DS * F_DS
    S = S[:a, :b].reshape(a // F_DS, F_DS, b // F_DS, F_DS, S.shape[2]).sum(axis=(1, 3))
    RES = RES * F_DS
print('stack', S.shape, 'res', RES, flush=True)
thr = ma.tissue_threshold_for(RES)
n = S.shape[2]


def mri_planes(aff_f):
    F = ants.from_numpy(np.asarray(nib.load(MRI_T).dataobj, np.float32), spacing=(0.25,) * 3)
    Mk = ants.from_numpy((np.asarray(nib.load(MRI_M).dataobj) > 0).astype(np.float32), spacing=(0.25,) * 3)
    grid = ants.from_numpy(np.zeros(S.shape, np.float32), spacing=(RES, RES, SL))
    m = ants.apply_transforms(grid, Mk, [aff_f], whichtoinvert=[True], interpolator='linear').numpy()
    return m


def slice_poses(mplanes):
    """Rigid 2D fit per slice: fixed = MRI mask plane, moving = histology mask; returns (theta_deg, dx_mm, dy_mm)
    as the displacement of the histology tissue centroid needed to match the MRI plane (fixed->moving inverted)."""
    out = np.full((n, 3), np.nan)
    for s in range(n):
        h = ndi.binary_fill_holes(S[..., s] > thr).astype(np.float32)
        m = (mplanes[..., s] > 0.5).astype(np.float32)
        if h.sum() < 20 or m.sum() < 20:
            continue
        fi = ants.from_numpy(ndi.gaussian_filter(m, 1.0), spacing=(RES, RES))
        mv = ants.from_numpy(ndi.gaussian_filter(h, 1.0), spacing=(RES, RES))
        r = ants.registration(fi, mv, 'Rigid', aff_metric='meansquares', aff_random_sampling_rate=1.0,
                              aff_iterations=(200, 100, 50), aff_shrink_factors=(4, 2, 1),
                              aff_smoothing_sigmas=(2, 1, 0), outprefix=f'{OUT}/tmp_')
        t = ants.read_transform(r['fwdtransforms'][0])  # maps fixed (MRI) -> moving (histology)
        p = np.array(t.parameters); c = np.array(t.fixed_parameters)
        A = p[:4].reshape(2, 2); b = p[4:6]
        cm = np.array(ndi.center_of_mass(m)) * RES                     # MRI tissue centre
        hc = A @ (cm - c) + b + c                                       # where it lands in the histology
        out[s] = [np.degrees(np.arctan2(A[1, 0], A[0, 0])), *(cm - hc)]  # histology -> MRI correction
    return out


def smooth(x, sigma):
    x = x.copy()
    good = ~np.isnan(x)
    x[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), x[good])
    x = ndi.median_filter(x, 5, mode='nearest')
    return ndi.gaussian_filter1d(x, sigma, mode='nearest')


def half(x, parity, sigma):
    y = np.full_like(x, np.nan); y[parity::2] = x[parity::2]
    return smooth(y, sigma)


fits = {}
for name, ref in [('fit1', 140), ('fit2', 260)]:
    aff, qc = ma.stack_to_mri_affine(S, RES, SL, MRI_T, MRI_M, f'{OUT}/{name}', ref_slice=ref, direction=1)
    poses = slice_poses(mri_planes(aff))
    fits[name] = dict(poses=poses, dice=qc['dice_affine'], scales=qc['scales_stack_axes'])
    print(f'{name}: ref {ref} dice {qc["dice_affine"]:.3f} scales {np.round(qc["scales_stack_axes"], 3)} '
          f'chosen {qc["chosen"]}', flush=True)
np.savez(f'{OUT}/poses.npz', **{k: v['poses'] for k, v in fits.items()})

labels = ['rotation (deg)', 'dx (mm)', 'dy (mm)']
summary = {}
for sigma in (10, 20):
    for k, lab in enumerate(labels):
        p1, p2 = fits['fit1']['poses'][:, k], fits['fit2']['poses'][:, k]
        s1, s2 = smooth(p1, sigma), smooth(p2, sigma)
        e, o = half(p1, 0, sigma), half(p1, 1, sigma)
        resid = p1 - smooth(p1, sigma)
        key = f'sigma{sigma} {lab}'
        summary[key] = dict(
            smooth_rms=float(np.sqrt(np.mean((s1 - s1.mean()) ** 2))),
            smooth_range=float(s1.max() - s1.min()),
            resid_rms=float(np.sqrt(np.nanmean(resid ** 2))),
            split_half_r=float(np.corrcoef(e, o)[0, 1]),
            split_half_rms_diff=float(np.sqrt(np.mean((e - o) ** 2))),
            cross_fit_r=float(np.corrcoef(s1, s2)[0, 1]),
            cross_fit_rms_diff=float(np.sqrt(np.mean(((s1 - s1.mean()) - (s2 - s2.mean())) ** 2))))
        print(key, {a: round(b, 3) for a, b in summary[key].items()}, flush=True)
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)

fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for k, lab in enumerate(labels):
    for name, c in [('fit1', 'tab:blue'), ('fit2', 'tab:orange')]:
        p = fits[name]['poses'][:, k]
        ax[k].plot(p, '.', color=c, ms=2, alpha=0.4)
        ax[k].plot(smooth(p, 10), color=c, label=f'{name} sigma 10')
    p1 = fits['fit1']['poses'][:, k]
    ax[k].plot(half(p1, 0, 10), 'k--', lw=0.8, label='fit1 even'); ax[k].plot(half(p1, 1, 10), 'k:', lw=0.8, label='fit1 odd')
    ax[k].set_ylabel(lab); ax[k].legend(ncol=4, fontsize=8)
    for x in (113, 252): ax[k].axvline(x, color='r', ls=':')
ax[2].set_xlabel('slice')
plt.tight_layout(); plt.savefig(f'{OUT}/poses.png', dpi=60)
