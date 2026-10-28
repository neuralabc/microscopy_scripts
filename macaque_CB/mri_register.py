"""Register one level of a slice-registration run to the D99 cerebellum template (affine, then optional nonlinear).

Point it at any run directory and any level (the final groupwise iteration is not always the one to use):
  uv run python macaque_CB/mri_register.py --run-dir <output_dir> --rescale 5 --level coreg12nl_win12_rigsyn_4_groupwise_iter9

Steps
  1. Affine (shear-free; mri_anchor.stack_to_mri_affine), one of:
     - --affine FILE: use this affine as it is (no fit);
     - a reference (default: mri_reference/zefir_stack_to_D99_reference.json if it exists): the fit starts from the
       reference affine and refines it;
     - no reference (--no-reference, or none saved): a full search, plus the in-plane 180 deg alternative of its
       result, refined the same way. The cerebellum outline (one hemisphere + vermis) is close to symmetric, and the
       full search gave a 180 deg turned result for one of four runs with almost the same Dice (2026-10-28). Both
       candidates are kept in the QC; the one with the better Dice is used and the choice is flagged when the
       Dice values are close. Check the QC images, then store a good result with --save-reference.
  2. Stability check against the reference (also for --affine): rotation difference and scale ratios. If a limit
     is passed the script stops (exit 2) before any output is written, unless --allow-unstable.
     For a full search, the orientation is chosen by MI, not by Dice (see mi_of).
  3. Nonlinear (unless --no-nonlinear): ANTs SyN (MI, smooth: flow sigma 3, gradient step 0.1) of the stack to D99,
     restricted to the region that the stack covers (its affine tissue mask, dilated, inside the dilated D99 mask),
     so that the missing hemisphere does not pull on the warp. D99 is a different animal: the nonlinear step imposes
     D99's shape and is for atlas mapping only; it does not change the slice reconstruction.

Outputs in <run_dir>/mri_registration/<level>/ (see README.md written there):
  stack_to_D99_affine.mat, stack_to_D99_warp.nii.gz, stack_to_D99_inverse_warp.nii.gz (ANTs)
  density_D99grid_affine.nii.gz, density_D99grid_nonlinear.nii.gz (cells/mm^3 on the D99 crop grid, D99 world affine)
  <subject>_<level>_density_mri.ome.zarr (tck_view; the stack voxels with the affine to D99 world, no resampling)
QC: <run_dir>/QC/mri/<level>/ (qc.json + images); the QC report (qc_report.py) is rebuilt to include them.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import nibabel
from scipy import ndimage as ndi

import env_compat  # noqa: F401  (threads, OOM priority, ANTsPy shims)
import ants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import mri_anchor  # noqa: E402
from evaluate_registration import MRI_T, MRI_M  # noqa: E402
from qc_report import find_levels, C1, C2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECT = 'zefir'
SLICE_MM = 0.05
DEFAULT_MISSING = [32, 59, 120, 160, 189, 228]
DEFAULT_REFERENCE = os.path.join(HERE, 'mri_reference', 'zefir_stack_to_D99_reference.json')


# ----------------------------------------------------------------------------------------------------------------
# affine helpers (ANTs AffineTransform: y = M (x - c) + t + c, maps MRI frame -> stack frame, both in mm)
# ----------------------------------------------------------------------------------------------------------------
def read_affine(f):
    t = ants.read_transform(f)
    p = np.array(t.parameters)
    return p[:9].reshape(3, 3), p[9:12], np.array(t.fixed_parameters)


def write_affine(M, t, c, f):
    ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform', dimension=3, matrix=M,
                                                    translation=t, center=tuple(c)), f)
    return f


def rotation_and_scales(f):
    """Polar rotation and singular values (sorted) of the 3x3 part."""
    M, _, _ = read_affine(f)
    U, s, Vt = np.linalg.svd(M)
    return U @ Vt, np.sort(s)


def rotation_difference_deg(f1, f2):
    R1, _ = rotation_and_scales(f1)
    R2, _ = rotation_and_scales(f2)
    R = R1.T @ R2
    if np.linalg.det(R) < 0:   # different handedness: no rotation maps one onto the other
        return 180.0
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def turn_in_plane_180(aff_f, out_f, stack_centre_mm):
    """The same placement turned by 180 deg in the section plane about the stack tissue centre:
    T'(x) = R (T(x) - p) + p with R = diag(-1, -1, 1) in the stack frame (i, j, section)."""
    M, t, c = read_affine(aff_f)
    R = np.diag([-1.0, -1.0, 1.0])
    p = np.asarray(stack_centre_mm, float)
    return write_affine(R @ M, R @ (t + c - p) + p - c, c, out_f)


# ----------------------------------------------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------------------------------------------
def planes(vol, centre):
    x, y, z = [int(round(v)) for v in centre]
    return [('sagittal x=%d' % x, vol[x, :, :].T), ('coronal y=%d' % y, vol[:, y, :].T), ('axial z=%d' % z, vol[:, :, z].T)]


def overlay_figure(T, W, thr, centre, title, out_png, extra=None):
    """D99 template (grey) with the stack (orange contour of its tissue, and its density in the right column)."""
    rows = [('D99 + stack outline', None), ('stack density in D99 grid', None)]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    tv = np.percentile(T[T > 0], 99) if (T > 0).any() else 1
    wv = np.percentile(W[W > thr], 99) if (W > thr).any() else 1
    for k, ((name, tp), (_, wp)) in enumerate(zip(planes(T, centre), planes(W, centre))):
        ax[0, k].imshow(tp, cmap='gray', origin='lower', vmin=0, vmax=tv)
        if (wp > thr).any():
            ax[0, k].contour(ndi.binary_fill_holes(wp > thr), [0.5], colors=C2, linewidths=1.2)
        ax[0, k].set_title(f'D99 {name}', loc='left', fontsize=9)
        ax[1, k].imshow(wp, cmap='gray', origin='lower', vmin=0, vmax=wv)
        ax[1, k].set_title(f'stack in D99 grid, {name}', loc='left', fontsize=9)
        for a in ax[:, k]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70, facecolor='white')
    plt.close(fig)


def jacobian_figure(J, region, centre, out_png):
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
    Jm = np.where(region, J, np.nan)
    for k, (name, p) in enumerate(planes(Jm, centre)):
        im = ax[k].imshow(p, cmap='RdBu_r', origin='lower', vmin=0.5, vmax=1.5)
        ax[k].set_title(f'Jacobian determinant, {name}', loc='left', fontsize=9)
        ax[k].set_xticks([]); ax[k].set_yticks([])
    fig.colorbar(im, ax=ax[2], fraction=0.05, pad=0.02)
    v = J[region]
    ax[3].hist(np.clip(v, 0, 3), bins=60, color=C1)
    ax[3].set_title('Jacobian in the registration region', loc='left', fontsize=9)
    ax[3].set_xlabel('det J (1 = no volume change)')
    fig.savefig(out_png, dpi=70, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--level', default=None, help='registration level (default: the last groupwise iteration)')
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--direction', type=int, choices=(1, -1), default=1,
                   help='D99 hemisphere for the stack (+1: slices run toward +x, right); only for a full search')
    p.add_argument('--reference', default=DEFAULT_REFERENCE, help='reference json (see --save-reference)')
    p.add_argument('--no-reference', action='store_true', help='full search, no reference and no stability check')
    p.add_argument('--affine', default=None, help='use this affine (ANTs .mat, MRI frame -> stack frame) as it is')
    p.add_argument('--max-rotation-deg', type=float, default=10.0)
    p.add_argument('--max-scale-change', type=float, default=0.25, help='largest allowed relative scale change')
    p.add_argument('--allow-unstable', action='store_true', help='go on (with a warning) when the check fails')
    p.add_argument('--no-nonlinear', action='store_true')
    p.add_argument('--syn-iterations', default='100,70,30')
    p.add_argument('--flow-sigma', type=float, default=3.0)
    p.add_argument('--grad-step', type=float, default=0.1)
    p.add_argument('--region-dilation-mm', type=float, default=1.5)
    p.add_argument('--no-zarr', action='store_true')
    p.add_argument('--missing', default=','.join(map(str, DEFAULT_MISSING)))
    p.add_argument('--save-reference', action='store_true',
                   help=f'store the affine used here as the reference ({DEFAULT_REFERENCE})')
    p.add_argument('--no-qc-report', action='store_true', help='do not rebuild QC/index.html')
    a = p.parse_args()

    run_dir = a.run_dir.rstrip('/') + '/'
    level = a.level or [lv for lv in find_levels(run_dir, a.subject) if 'groupwise_iter' in lv][-1]
    out = os.path.join(run_dir, 'mri_registration', level)
    qc_dir = os.path.join(run_dir, 'QC', 'mri', level)
    work = os.path.join(out, 'work')
    for d in (out, qc_dir, work):
        os.makedirs(d, exist_ok=True)
    cmd = 'mri_register.py ' + ' '.join(sys.argv[1:])
    res_mm = a.rescale * 10 / 1000.0
    thr = mri_anchor.tissue_threshold_for(res_mm)
    voxel_mm3 = res_mm * res_mm * SLICE_MM
    qc = dict(level=level, run_dir=run_dir, command=cmd, time=time.strftime('%Y-%m-%d %H:%M'), status='running',
              tables={}, images=[], notes=[])

    def write_qc():
        with open(os.path.join(qc_dir, 'qc.json'), 'w') as fh:
            json.dump(qc, fh, indent=1, default=float)

    def rebuild_report():
        if a.no_qc_report:
            return
        subprocess.run([sys.executable, os.path.join(HERE, 'qc_report.py'), '--run-dir', run_dir,
                        '--rescale', str(a.rescale), '--subject', a.subject, '--missing', a.missing], check=False)

    stack = np.asarray(nibabel.load(os.path.join(run_dir, f'{a.subject}_{level}_stack.nii.gz')).dataobj, dtype=np.float32)
    Tn = nibabel.load(MRI_T)
    T, Mmask, mri_sp, _ = mri_anchor.load_mri(MRI_T, MRI_M)
    F = ants.from_numpy(T, spacing=(mri_sp,) * 3)
    Mv = mri_anchor.downsample_stack(stack, res_mm, SLICE_MM, mri_sp)          # stack frame, mri_sp isotropic
    tissue = np.stack([ndi.binary_fill_holes(stack[..., s] > thr) for s in range(stack.shape[2])], axis=2)
    centre_mm = np.array(ndi.center_of_mass(tissue)) * np.array([res_mm, res_mm, SLICE_MM])
    del tissue
    thr_mv = thr  # Mv holds counts per stack voxel (smoothed), so the same threshold applies

    def dice_of(aff_f):
        """Dice of the affine-placed stack tissue with the D99 mask, inside the left-right range that the stack
        covers (the sections hold one hemisphere + vermis, so the other hemisphere is left out)."""
        W = ants.apply_transforms(F, Mv, [aff_f]).numpy()
        xs = np.flatnonzero((W > thr_mv).any(axis=(1, 2)))
        region = np.zeros_like(Mmask)
        if len(xs):
            region[xs.min():xs.max() + 1] = True
        return mri_anchor._dice(W, Mmask, region, thr_mv), W

    mi_region = ants.from_numpy(ndi.binary_dilation(Mmask, iterations=max(1, int(round(1.0 / mri_sp)))).astype(np.float32),
                                spacing=(mri_sp,) * 3)

    def mi_of(aff_f):
        """Mattes MI (ANTs sign: lower is better) of the affine-placed stack with D99, in the dilated D99 mask.
        It tells the two in-plane orientations apart where Dice does not (J20, 2026-10-28: MI -0.092 for the
        anatomically correct one, -0.058 for the one turned by 180 deg; Dice 0.838 vs 0.839)."""
        return float(ants.image_similarity(F, ants.apply_transforms(F, Mv, [aff_f]),
                                           metric_type='MattesMutualInformation', fixed_mask=mi_region))

    # ---------------------------------------------------------------- 1. affine
    ref = None
    if not a.no_reference and os.path.isfile(a.reference):   # also for --affine: the stability check uses it
        with open(a.reference) as fh:
            ref = json.load(fh)
        ref['affine_file'] = os.path.join(os.path.dirname(a.reference), ref['affine_file'])
    cands = {}
    if a.affine:
        aff_f = shutil.copyfile(a.affine, os.path.join(work, 'given_affine.mat'))
        d, _ = dice_of(aff_f)
        cands['given'] = dict(file=aff_f, dice=d)
        chosen = 'given'
        qc['notes'].append(f'affine given: {a.affine} (no fit)')
    elif ref is not None:
        f_ref, q_ref = mri_anchor.stack_to_mri_affine(stack, res_mm, SLICE_MM, MRI_T, MRI_M, os.path.join(work, 'from_reference'),
                                                      init_state=dict(affine_file=ref['affine_file'],
                                                                      direction=ref['direction'], x0_mm=ref['x0_mm']))
        d0, _ = dice_of(ref['affine_file'])
        cands['reference (unrefined)'] = dict(file=ref['affine_file'], dice=d0)
        cands['refined from reference'] = dict(file=f_ref, dice=dice_of(f_ref)[0], direction=q_ref['direction'],
                                               x0_mm=q_ref['x0_mm'])
        chosen = 'refined from reference'
        qc['notes'].append(f'started from the reference {a.reference} ({ref.get("source", "")})')
    else:
        f1, q1 = mri_anchor.stack_to_mri_affine(stack, res_mm, SLICE_MM, MRI_T, MRI_M, os.path.join(work, 'search'),
                                                direction=a.direction)
        cands['full search'] = dict(file=f1, dice=dice_of(f1)[0], direction=q1['direction'], x0_mm=q1['x0_mm'])
        f_alt = turn_in_plane_180(f1, os.path.join(work, 'turned_180_init.mat'), centre_mm)
        f2, q2 = mri_anchor.stack_to_mri_affine(stack, res_mm, SLICE_MM, MRI_T, MRI_M, os.path.join(work, 'turned_180'),
                                                init_state=dict(affine_file=f_alt, direction=q1['direction'],
                                                                x0_mm=q1['x0_mm']))
        cands['turned 180 deg in-plane'] = dict(file=f2, dice=dice_of(f2)[0], direction=q2['direction'],
                                                x0_mm=q2['x0_mm'])
        for v in cands.values():
            v['mi'] = mi_of(v['file'])
        # the outline (Dice) cannot tell the two orientations apart; the intensities (MI) can
        chosen = min(cands, key=lambda k: cands[k]['mi'])
        other = [k for k in cands if k != chosen][0]
        rel = (cands[other]['mi'] - cands[chosen]['mi']) / abs(cands[chosen]['mi'])
        qc['notes'].append(f'full search (no reference): chosen "{chosen}" by MI ({cands[chosen]["mi"]:.4f} vs '
                           f'{cands[other]["mi"]:.4f} for "{other}"; Dice {cands[chosen]["dice"]:.3f} vs '
                           f'{cands[other]["dice"]:.3f}; rotation between them '
                           f'{rotation_difference_deg(cands[chosen]["file"], cands[other]["file"]):.0f} deg)')
        if rel < 0.2:
            qc['notes'].append('AMBIGUOUS: the two in-plane orientations have a similar MI. Check the images below '
                               '(folia dorsal and posterior, brainstem and peduncles ventral and anterior) before using it.')
    aff_f = shutil.copyfile(cands[chosen]['file'], os.path.join(out, 'stack_to_D99_affine.mat'))
    rot, scales = rotation_and_scales(aff_f)
    aff_tab = {f'Dice with D99 in the stack left-right range ({k})': v['dice'] for k, v in cands.items()}
    for k, v in cands.items():
        aff_tab[f'Mattes MI, lower is better ({k})'] = v.get('mi', mi_of(v['file']))
    aff_tab['chosen'] = chosen
    aff_tab['scales MRI->stack (sorted singular values)'] = ', '.join(f'{s:.3f}' for s in scales)
    for k, v in cands.items():
        _, W = dice_of(v['file'])
        png = f'affine_{k.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        overlay_figure(T, W, thr_mv, np.array(ndi.center_of_mass(W > thr_mv)), f'affine: {k} (Dice {v["dice"]:.3f})',
                       os.path.join(qc_dir, png))
        qc['images'].append(dict(file=png, caption=f'affine candidate "{k}"' + (' (used)' if k == chosen else '')))

    # ---------------------------------------------------------------- 2. stability check
    if ref is not None:
        rd = rotation_difference_deg(ref['affine_file'], aff_f)
        _, s_ref = rotation_and_scales(ref['affine_file'])
        sc = float(np.max(np.abs(scales / s_ref - 1)))
        aff_tab['rotation vs reference (deg)'] = rd
        aff_tab['largest scale change vs reference'] = sc
        problems = []
        if rd > a.max_rotation_deg:
            problems.append(f'rotation {rd:.1f} deg > {a.max_rotation_deg}')
        if sc > a.max_scale_change:
            problems.append(f'scale change {sc:.2f} > {a.max_scale_change}')
        if problems:
            qc['tables']['affine'] = aff_tab
            msg = 'stability check failed: ' + '; '.join(problems)
            if not a.allow_unstable:
                qc['status'] = 'FAILED: ' + msg
                os.remove(aff_f)
                write_qc(); rebuild_report()
                print(qc['status'], file=sys.stderr)
                sys.exit(2)
            qc['notes'].append('WARNING: ' + msg + ' (--allow-unstable)')
    qc['tables']['affine'] = aff_tab
    write_qc()

    # density on the D99 grid (counts per stack voxel -> cells/mm^3), affine
    W_aff = ants.apply_transforms(F, Mv, [aff_f])
    nibabel.save(nibabel.Nifti1Image(W_aff.numpy() / voxel_mm3, Tn.affine), os.path.join(out, 'density_D99grid_affine.nii.gz'))

    # ---------------------------------------------------------------- 3. nonlinear
    if not a.no_nonlinear:
        Wa = W_aff.numpy()
        hm = np.array([ndi.binary_fill_holes(p) for p in (Wa > thr_mv)])
        it = max(1, int(round(a.region_dilation_mm / mri_sp)))
        region = ndi.binary_dilation(hm, iterations=it) & ndi.binary_dilation(Mmask, iterations=it)
        Fm = ants.from_numpy(region.astype(np.float32), spacing=(mri_sp,) * 3)
        seed_before = os.environ.get('ANTS_RANDOM_SEED')
        os.environ['ANTS_RANDOM_SEED'] = '1'
        try:
            r = ants.registration(F, Mv, 'SyNOnly', initial_transform=aff_f, mask=Fm, syn_metric='mattes',
                                  syn_sampling=32, reg_iterations=tuple(int(v) for v in a.syn_iterations.split(',')),
                                  flow_sigma=a.flow_sigma, total_sigma=0, grad_step=a.grad_step,
                                  outprefix=os.path.join(work, 'syn_'))
        finally:
            if seed_before is None:
                os.environ.pop('ANTS_RANDOM_SEED', None)
            else:
                os.environ['ANTS_RANDOM_SEED'] = seed_before
        warp_f = [f for f in r['fwdtransforms'] if f.endswith('Warp.nii.gz') and 'Inverse' not in f][0]
        iwarp_f = [f for f in r['invtransforms'] if f.endswith('InverseWarp.nii.gz')][0]
        warp_out = shutil.copyfile(warp_f, os.path.join(out, 'stack_to_D99_warp.nii.gz'))
        iwarp_out = shutil.copyfile(iwarp_f, os.path.join(out, 'stack_to_D99_inverse_warp.nii.gz'))
        W_nl = ants.apply_transforms(F, Mv, [warp_out, aff_f])
        nibabel.save(nibabel.Nifti1Image(W_nl.numpy() / voxel_mm3, Tn.affine),
                     os.path.join(out, 'density_D99grid_nonlinear.nii.gz'))
        Wn = W_nl.numpy()
        disp = ants.image_read(warp_out).numpy()
        dmag = np.linalg.norm(disp, axis=-1)[region]
        J = ants.create_jacobian_determinant_image(F, warp_out).numpy()
        jr = J[region]
        mi = lambda W: float(ants.image_similarity(F, ants.from_numpy(W, spacing=(mri_sp,) * 3),
                                                   metric_type='MattesMutualInformation', fixed_mask=Fm))
        qc['tables']['nonlinear'] = {
            'Dice in the SyN region (not comparable with the affine table), affine': mri_anchor._dice(Wa, Mmask, region, thr_mv),
            'Dice in the SyN region, nonlinear': mri_anchor._dice(Wn, Mmask, region, thr_mv),
            'Mattes MI in region, affine (lower is better)': mi(Wa),
            'Mattes MI in region, nonlinear': mi(Wn),
            'displacement in region, median (mm)': float(np.median(dmag)),
            'displacement in region, 95th pct (mm)': float(np.percentile(dmag, 95)),
            'displacement in region, max (mm)': float(dmag.max()),
            'Jacobian det, 5th pct': float(np.percentile(jr, 5)),
            'Jacobian det, 95th pct': float(np.percentile(jr, 95)),
            'Jacobian det <= 0 (folding), fraction': float((jr <= 0).mean()),
            'SyN': f'iterations {a.syn_iterations}, flow sigma {a.flow_sigma}, gradient step {a.grad_step}, MI, '
                   f'region = stack tissue (affine) dilated {a.region_dilation_mm} mm, inside the dilated D99 mask'}
        c = np.array(ndi.center_of_mass(Wn > thr_mv))
        overlay_figure(T, Wn, thr_mv, c, 'nonlinear (affine + SyN)', os.path.join(qc_dir, 'nonlinear.png'))
        jacobian_figure(J, region, c, os.path.join(qc_dir, 'jacobian.png'))
        qc['images'] += [dict(file='nonlinear.png', caption='nonlinear: D99 with the warped stack outline'),
                         dict(file='jacobian.png', caption='Jacobian determinant of the warp (blue < 1: the stack '
                                                           'is compressed to fit D99; red > 1: stretched)')]
        write_qc()

    # ---------------------------------------------------------------- outputs
    readme = f"""# MRI registration of {level}

Written {qc['time']} by `{cmd}`.

Frames (both in mm, index * spacing, zero origin, identity direction; as in mri_anchor.py):
- stack frame: `ants.from_numpy(stack, spacing=({res_mm}, {res_mm}, {SLICE_MM}))` of `{a.subject}_{level}_stack.nii.gz`
- D99 frame: `ants.from_numpy(D99_template, spacing=({mri_sp},) * 3)` of `{os.path.basename(MRI_T)}`
  (the world (RAS) affine of that file maps it to D99 world coordinates).

Transforms (ANTs; they map D99 frame points to stack frame points, so they resample stack images into D99):
- `stack_to_D99_affine.mat`: shear-free affine
- `stack_to_D99_warp.nii.gz` / `stack_to_D99_inverse_warp.nii.gz`: SyN after the affine (if run)

Stack image -> D99 grid:
    ants.apply_transforms(fixed=D99, moving=stack_img, transformlist=['stack_to_D99_warp.nii.gz', 'stack_to_D99_affine.mat'])
D99 image (e.g. labels) -> stack grid:
    ants.apply_transforms(fixed=stack_img, moving=D99_img, transformlist=['stack_to_D99_affine.mat', 'stack_to_D99_inverse_warp.nii.gz'],
                          whichtoinvert=[True, False], interpolator='genericLabel')
Points, stack frame mm -> D99 frame mm (point transforms go the opposite way from image transforms):
    ants.apply_transforms_to_points(3, points_df, ['stack_to_D99_affine.mat', 'stack_to_D99_inverse_warp.nii.gz'],
                                    whichtoinvert=[True, False])

Densities on the D99 grid (cells/mm^3, the D99 world affine): `density_D99grid_affine.nii.gz`,
`density_D99grid_nonlinear.nii.gz`. The stack is smoothed to {mri_sp} mm before resampling.
QC: `{os.path.relpath(qc_dir, run_dir)}` and `QC/index.html`.
"""
    with open(os.path.join(out, 'README.md'), 'w') as fh:
        fh.write(readme)

    if not a.no_zarr:
        zarr_out = os.path.join(run_dir, f'{a.subject}_{level}_density_mri.ome.zarr')
        r = subprocess.run([sys.executable, os.path.join(HERE, 'export_ome_zarr.py'), '--run-dir', run_dir,
                            '--rescale', str(a.rescale), '--level', level, '--placement', 'mri', '--out', zarr_out,
                            '--mri-affine', aff_f, '--missing', a.missing, '--overwrite'], check=False)
        qc['notes'].append(f'OME-Zarr (mri placement, this affine): {zarr_out}' if r.returncode == 0
                           else f'OME-Zarr export failed (exit {r.returncode})')

    if a.save_reference and cands[chosen].get('x0_mm') is None:
        qc['notes'].append('not saved as the reference: a given affine has no slice position (x0_mm); fit it instead')
    elif a.save_reference:
        rdir = os.path.dirname(DEFAULT_REFERENCE)
        os.makedirs(rdir, exist_ok=True)
        shutil.copyfile(aff_f, os.path.join(rdir, 'zefir_stack_to_D99_reference_affine.mat'))
        info = cands[chosen]
        with open(DEFAULT_REFERENCE, 'w') as fh:
            json.dump(dict(affine_file='zefir_stack_to_D99_reference_affine.mat', direction=info.get('direction', a.direction),
                           x0_mm=info.get('x0_mm'), source=f'{run_dir} {level}', rescale=a.rescale, dice=info['dice'],
                           saved=qc['time']), fh, indent=1)
        qc['notes'].append(f'saved as the reference: {DEFAULT_REFERENCE}')

    qc['status'] = 'ok'
    write_qc()
    rebuild_report()
    print(json.dumps(qc['tables'], indent=1, default=float))


if __name__ == '__main__':
    main()
