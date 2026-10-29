"""Pull the D99 atlas labels (lobules, deep nuclei, brainstem) into the stack grid of one registered level.

WORK IN PROGRESS (2026-10-28): a first atlas-propagation version. The lobule boundaries, the vermis rule and the
lateral coverage are not validated; see LOBULE_LABELS_HANDOFF.md before using the labels for analysis.

Run mri_register.py on the level first (it runs this script itself at its end); for another level or settings:
  uv run python macaque_CB/atlas_labels.py --run-dir <output_dir> --rescale 5 --level <level>

Labels: D99 v2.0 (Saleem et al. 2021), local_extras/cerebellum_target/D99_cerebellum_brainstem_labels_crop.nii.gz
(the grid of the template that mri_register.py fits), names in mri_reference/D99_v2.0_labels_cerebellum_brainstem.tsv.
D99 labels grey matter structures only: the cerebellar white matter and much of the brainstem (e.g. the reticular
formation) have no label. The D99 region map (D99_cerebellum_brainstem_regions_crop: 1 cerebellum, 2 brainstem)
separates these two.
Transforms: <run_dir>/mri_registration/<level>/ (affine + inverse SyN warp; the affine alone if there is no warp),
nearest-label (ANTs genericLabel) resampling onto the full-resolution stack grid, in slabs of sections.

Vermis: D99 has no vermis label. The posterior lobe has separate labels for the vermis (lobules VI-X, 113-117) and
the hemisphere (simple lobule, crus I/II, paramedian lobule, paraflocculus, flocculus), but lobules I-V (108-112)
run from the midline to the lateral edge. Here a voxel of lobules I-V is vermis if |D99 x| <= the anterior vermis
half-width (default 4.0 mm: the median half-width of D99 lobule VI, the vermal lobule next to the anterior lobe;
lobules VI-X: median 2.5 mm, 90th percentile 4.25 mm). The D99 x map is written too, so that another rule can be
used later without a new run. Left/right is that of the D99 placement (the reference affine puts the sections on
D99's right side); the true side of the zefir hemisphere is not known from the registration.

Outputs beside the stack (same grid, affine and header as <subject>_<level>_stack.nii.gz):
  <subject>_<level>_D99_labels.nii.gz   uint8, the D99 label value in tissue (tissue_mask.py), else 0, and:
                                          252 = D99 cerebellum without a label (white matter in D99; in zefir also
                                                some grey matter at the label edges)
                                          253 = D99 brainstem without a label (e.g. reticular formation)
                                          254 = inside the D99 mask but in neither region (rare)
                                          255 = tissue outside the D99 mask (the atlas does not cover it: brainstem
                                                cut by the D99 crop, and a rim where the shapes differ)
                                        tissue outside the D99 mask that is at most --fill-mm (default 1.0) from a D99
                                        label takes the nearest label (the rim; 0 turns this off)
  <subject>_<level>_D99_regions.nii.gz  uint8: 1 vermis cortex, 2 hemisphere cortex, 3 deep cerebellar nuclei,
                                        4 other D99 label (brainstem nuclei, fibre bundles), 252-255 as above
  <subject>_<level>_D99_x_mm.nii.gz     float32, D99 world x (mm; 0 = D99 midline, + = right), NaN outside the D99 grid
  <subject>_<level>_D99_labels.tsv      per label: voxels, volume, cells and density (measured sections only; the
                                        interpolated sections are left out), cells in the vermis, cells left/right
                                        if count_preserving_stack.py has run for the level: also cells_cp (sum of the
                                        count-preserving stack, i.e. cells without the warp's area change),
                                        native_volume_mm3 (native tissue volume) and density_cp_cells_per_mm3 = their
                                        ratio; the first columns are unchanged (sum of the pipeline stack, final-frame
                                        volume)
QC: <run_dir>/QC/mri/<level>/atlas_labels.json + images (the QC report shows them with the MRI registration).

Accuracy: D99 is a different animal at 0.25 mm, and the SyN is smooth (flow sigma 3, median displacement ~0.7 mm
at rescale 20), so boundaries between lobules can be off by about one folium. Check the QC images.
"""
import argparse
import json
import os
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

from evaluate_registration import MRI_DIR, MRI_M  # noqa: E402
from tissue_mask import tissue_mask  # noqa: E402
from qc_report import C1, C2, C3, find_levels  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECT = 'zefir'
SLICE_MM = 0.05
DEFAULT_MISSING = [32, 59, 120, 160, 189, 228]
D99_LABELS = os.path.join(MRI_DIR, 'D99_cerebellum_brainstem_labels_crop.nii.gz')
D99_REGIONS = os.path.join(MRI_DIR, 'D99_cerebellum_brainstem_regions_crop.nii.gz')   # 1 cerebellum, 2 brainstem
D99_NAMES = os.path.join(HERE, 'mri_reference', 'D99_v2.0_labels_cerebellum_brainstem.tsv')

CB_NO_LABEL, BS_NO_LABEL, NO_LABEL, OUTSIDE_ATLAS = 252, 253, 254, 255
SPECIAL = (CB_NO_LABEL, BS_NO_LABEL, NO_LABEL, OUTSIDE_ATLAS)
ANTERIOR_LOBE = [108, 109, 110, 111, 112]                       # I-V: midline to lateral edge
VERMIS_POSTERIOR = [113, 114, 115, 116, 117]                    # VI-X (vermal parts)
HEMISPHERE = [118, 119, 120, 121, 123, 125, 222, 223, 224, 225]  # Par_p/a, Cr_II/I, lPFl, Fl, Sim_a/p, dPFl, vPFl
NUCLEI = [61, 62, 63, 64]                                       # DN, AIN, PIN, FN
REGION_NAMES = {1: 'vermis cortex', 2: 'hemisphere cortex', 3: 'deep cerebellar nuclei',
                4: 'other D99 label (brainstem, fibre bundles)', CB_NO_LABEL: 'D99 cerebellum, no label (white matter)',
                BS_NO_LABEL: 'D99 brainstem, no label', NO_LABEL: 'in D99 mask, no region',
                OUTSIDE_ATLAS: 'outside the D99 mask'}


def load_names(f=D99_NAMES):
    names = {}
    with open(f) as fh:
        for line in fh:
            if line.startswith('#') or line.startswith('value'):
                continue
            v, ab, name, group, sub = (line.rstrip('\n').split('\t') + [''] * 5)[:5]
            names[int(v)] = dict(abbreviation=ab, name=name, group=group, subgroup=sub)
    return names


def pull_back(moving, shape, res_mm, transforms, invert, interpolator, slab, default=0.0):
    """Resample a D99-grid array onto the stack grid (index * spacing frames, as in mri_register.py), slab by slab."""
    mv = ants.from_numpy(np.ascontiguousarray(moving, dtype=np.float32), spacing=(0.25,) * 3)
    out = np.zeros(shape, np.float32)
    for s0 in range(0, shape[2], slab):
        s1 = min(shape[2], s0 + slab)
        fx = ants.from_numpy(np.zeros(shape[:2] + (s1 - s0,), np.float32), spacing=(res_mm, res_mm, SLICE_MM),
                             origin=(0.0, 0.0, s0 * SLICE_MM))
        out[..., s0:s1] = ants.apply_transforms(fx, mv, transforms, whichtoinvert=invert, interpolator=interpolator,
                                                defaultvalue=default).numpy()
    return out


def fill_from_nearest(lab, res_mm, fill_mm, slab):
    """Tissue outside the D99 mask (OUTSIDE_ATLAS) within fill_mm of a D99 label takes the nearest label."""
    out = lab.copy()
    n, m = lab.shape[2], int(np.ceil(fill_mm / SLICE_MM))
    for s0 in range(0, n, slab):
        s1 = min(n, s0 + slab)
        a, b = max(0, s0 - m), min(n, s1 + m)
        sub = lab[..., a:b]
        if not (sub[..., s0 - a:s1 - a] == OUTSIDE_ATLAS).any():
            continue
        src = (sub > 0) & (sub < CB_NO_LABEL)
        if not src.any():
            continue
        d, idx = ndi.distance_transform_edt(~src, sampling=(res_mm, res_mm, SLICE_MM), return_indices=True)
        take = (sub == OUTSIDE_ATLAS) & (d <= fill_mm)
        near = sub[tuple(idx)]
        blk, t = out[..., s0:s1], take[..., s0 - a:s1 - a]
        blk[t] = near[..., s0 - a:s1 - a][t]
    return out


def regions_of(lab, x_mm, anterior_halfwidth_mm):
    r = np.zeros(lab.shape, np.uint8)
    r[np.isin(lab, HEMISPHERE)] = 2
    r[np.isin(lab, VERMIS_POSTERIOR)] = 1
    ant = np.isin(lab, ANTERIOR_LOBE)
    r[ant] = np.where(np.abs(x_mm[ant]) <= anterior_halfwidth_mm, 1, 2)
    r[np.isin(lab, NUCLEI)] = 3
    r[(lab > 0) & (lab < CB_NO_LABEL) & (r == 0)] = 4
    for v in SPECIAL:
        r[lab == v] = v
    return r


def save_like(arr, ref_img, dtype, descrip, out):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    hdr['descrip'] = descrip.encode('ascii')[:79]
    nibabel.save(nibabel.Nifti1Image(arr.astype(dtype), ref_img.affine, hdr), out)
    return out


def label_figure(counts, lab, reg, res_mm, slices, names, out_png):
    """Sagittal sections: density (grey), vermis / hemisphere / nuclei (colour), D99 label edges (white), names."""
    tissue = lab > 0
    vmax = np.percentile(counts[tissue], 99) if tissue.any() else 1
    cols = {1: C1, 2: C2, 3: C3}
    nc = 4
    nr = int(np.ceil(len(slices) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.6 * nr), squeeze=False)
    for a in ax.flat:
        a.axis('off')
    ext = [0, counts.shape[0] * res_mm, 0, counts.shape[1] * res_mm]
    for a, s in zip(ax.flat, slices):
        a.imshow(counts[..., s].T, cmap='gray', origin='lower', vmin=0, vmax=vmax, extent=ext)
        rgba = np.zeros(counts.shape[:2] + (4,))
        for k, c in cols.items():
            rgba[reg[..., s] == k] = matplotlib.colors.to_rgba(c, 0.35)
        a.imshow(np.transpose(rgba, (1, 0, 2)), origin='lower', extent=ext)
        L = lab[..., s].astype(int)
        edge = ((L != np.roll(L, 1, 0)) | (L != np.roll(L, 1, 1))) & (L > 0) & (L < CB_NO_LABEL)
        ee = np.zeros(edge.shape + (4,))
        ee[edge] = (1, 1, 1, 0.8)
        a.imshow(np.transpose(ee, (1, 0, 2)), origin='lower', extent=ext)
        for v in np.unique(L[(L > 0) & (L < CB_NO_LABEL)]):
            m = L == v
            if m.sum() * res_mm ** 2 < 0.6:
                continue
            ci, cj = ndi.center_of_mass(m)
            a.text(ci * res_mm, cj * res_mm, names.get(int(v), {}).get('abbreviation', str(v)), fontsize=7,
                   color='white', ha='center', va='center')
        bb = np.argwhere(counts[..., s] > 0)
        if len(bb):
            lo, hi = bb.min(0) * res_mm - 1, bb.max(0) * res_mm + 1
            a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
        a.set_title(f'section {s}', loc='left', fontsize=9)
    fig.suptitle('D99 labels on the stack: blue vermis, orange hemisphere, green deep nuclei; white: label edges',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70, facecolor='white')
    plt.close(fig)


def across_sections_figure(counts, reg, res_mm, out_png):
    """Views across the sections (rows: in-plane position, columns: section), where the vermis is a band."""
    tissue = reg > 0
    vmax = np.percentile(counts[tissue], 99) if tissue.any() else 1
    ii, jj = np.nonzero(tissue.any(2))
    rows = [('in-plane row j=%d' % j, counts[:, j, :], reg[:, j, :]) for j in np.percentile(jj, [40, 60]).astype(int)]
    fig, ax = plt.subplots(1, len(rows), figsize=(8 * len(rows), 4.2), squeeze=False)
    for a, (name, c, r) in zip(ax[0], rows):
        ext = [0, c.shape[1] * SLICE_MM, 0, c.shape[0] * res_mm]
        a.imshow(c, cmap='gray', origin='lower', vmin=0, vmax=vmax, extent=ext, aspect='equal')
        rgba = np.zeros(c.shape + (4,))
        for k, col in {1: C1, 2: C2, 3: C3}.items():
            rgba[r == k] = matplotlib.colors.to_rgba(col, 0.35)
        a.imshow(rgba, origin='lower', extent=ext, aspect='equal')
        a.set_title(f'across sections, {name}', loc='left', fontsize=9)
        a.set_xlabel('section position (mm)'); a.set_ylabel('in-plane i (mm)')
        a.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--level', default=None, help='registration level (default: the last one with an MRI registration)')
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--missing', default=','.join(map(str, DEFAULT_MISSING)))
    p.add_argument('--anterior-vermis-halfwidth-mm', type=float, default=4.0,
                   help='lobules I-V: vermis if |D99 x| <= this (mm)')
    p.add_argument('--fill-mm', type=float, default=1.0,
                   help='tissue outside the D99 mask within this distance of a label takes the nearest label (0: off)')
    p.add_argument('--slab', type=int, default=32, help='sections per resampling slab (memory)')
    p.add_argument('--no-qc-report', action='store_true', help='do not rebuild QC/index.html')
    a = p.parse_args()

    run_dir = a.run_dir.rstrip('/') + '/'
    if a.level is None:   # the last level (pipeline order) with an MRI registration
        a.level = [lv for lv in find_levels(run_dir, a.subject)
                   if os.path.isfile(os.path.join(run_dir, 'mri_registration', lv, 'stack_to_D99_affine.mat'))][-1]
    reg_dir = os.path.join(run_dir, 'mri_registration', a.level)
    qc_dir = os.path.join(run_dir, 'QC', 'mri', a.level)
    os.makedirs(qc_dir, exist_ok=True)
    cmd = 'atlas_labels.py ' + ' '.join(sys.argv[1:])
    res_mm = a.rescale * 10 / 1000.0
    voxel_mm3 = res_mm * res_mm * SLICE_MM
    missing = [int(v) for v in a.missing.split(',') if v.strip()]
    names = load_names()

    stack_img = nibabel.load(os.path.join(run_dir, f'{a.subject}_{a.level}_stack.nii.gz'))
    counts = np.asarray(stack_img.dataobj, dtype=np.float32)
    shape = counts.shape
    aff_f = os.path.join(reg_dir, 'stack_to_D99_affine.mat')
    iwarp_f = os.path.join(reg_dir, 'stack_to_D99_inverse_warp.nii.gz')
    if os.path.isfile(iwarp_f):
        tl, inv, used = [aff_f, iwarp_f], [True, False], 'affine + inverse SyN warp'
    else:
        tl, inv, used = [aff_f], [True], 'affine only (no SyN warp found)'

    Ln = nibabel.load(D99_LABELS)
    L = np.asarray(Ln.dataobj).astype(np.float32)
    if not np.allclose(Ln.affine[:3, :3], np.diag([0.25] * 3)):
        raise ValueError(f'{D99_LABELS}: expected 0.25 mm RAS voxels without rotation')
    mask = np.asarray(nibabel.load(MRI_M).dataobj) > 0
    R = np.asarray(nibabel.load(D99_REGIONS).dataobj).astype(np.float32)
    if mask.shape != L.shape or R.shape != L.shape:
        raise ValueError('D99 labels, mask and regions grids differ')
    xw = (np.arange(L.shape[0]) * Ln.affine[0, 0] + Ln.affine[0, 3]).astype(np.float32)
    X = np.broadcast_to(xw[:, None, None], L.shape)

    tissue = tissue_mask(counts, res_mm).astype(bool)
    lab = pull_back(L, shape, res_mm, tl, inv, 'genericLabel', a.slab).round().astype(np.uint8)
    inmask = pull_back(mask, shape, res_mm, tl, inv, 'genericLabel', a.slab) > 0.5
    region = pull_back(R, shape, res_mm, tl, inv, 'genericLabel', a.slab).round().astype(np.uint8)
    x_mm = pull_back(X, shape, res_mm, tl, inv, 'linear', a.slab, default=-999.0)
    x_mm[x_mm < -900] = np.nan
    lab[~tissue] = 0
    lab[tissue & (lab == 0) & inmask & (region == 1)] = CB_NO_LABEL
    lab[tissue & (lab == 0) & inmask & (region == 2)] = BS_NO_LABEL
    lab[tissue & (lab == 0) & inmask] = NO_LABEL
    del region
    lab[tissue & (lab == 0) & ~inmask] = OUTSIDE_ATLAS
    exact_outside = lab == OUTSIDE_ATLAS
    if a.fill_mm > 0:
        lab = fill_from_nearest(lab, res_mm, a.fill_mm, a.slab)
    filled = exact_outside & (lab != OUTSIDE_ATLAS)
    reg = regions_of(lab, np.nan_to_num(x_mm, nan=1e3), a.anterior_vermis_halfwidth_mm)

    cp_f = os.path.join(run_dir, f'{a.subject}_{a.level}_counts_stack.nii.gz')   # count_preserving_stack.py
    area_f = os.path.join(run_dir, f'{a.subject}_{a.level}_native_area_stack.nii.gz')
    has_cp = os.path.isfile(cp_f) and os.path.isfile(area_f)
    pre = os.path.join(run_dir, f'{a.subject}_{a.level}_D99')
    d = f'D99 v2.0; 252 cb/253 bs no label, 255 outside D99; fill {a.fill_mm} mm'
    save_like(lab, stack_img, np.uint8, d, pre + '_labels.nii.gz')
    save_like(reg, stack_img, np.uint8, f'1 vermis 2 hemisphere 3 nuclei 4 other; I-V vermis |x|<={a.anterior_vermis_halfwidth_mm}mm',
              pre + '_regions.nii.gz')
    save_like(x_mm, stack_img, np.float32, 'D99 world x (mm), 0 = D99 midline', pre + '_x_mm.nii.gz')

    # ---------------------------------------------------------------- table (measured sections only)
    meas = np.ones(shape[2], bool)
    meas[[s for s in missing if s < shape[2]]] = False
    lm, cm, rm, xm = lab[..., meas], counts[..., meas], reg[..., meas], x_mm[..., meas]
    cells_tissue = float(cm[lm > 0].sum())
    if has_cp:
        cpm = np.asarray(nibabel.load(cp_f).dataobj, dtype=np.float32)[..., meas]
        arm = np.asarray(nibabel.load(area_f).dataobj, dtype=np.float32)[..., meas]
    rows = []
    for v in np.unique(lm[lm > 0]):
        sel = lm == v
        c = cm[sel]
        nm = names.get(int(v), dict(abbreviation={CB_NO_LABEL: 'cb_none', BS_NO_LABEL: 'bs_none', NO_LABEL: 'none',
                                                  OUTSIDE_ATLAS: 'outside'}.get(int(v), str(v)),
                                    name=REGION_NAMES.get(int(v), ''), group='', subgroup=''))
        xs = xm[sel]
        rows.append(dict(value=int(v), abbreviation=nm['abbreviation'], name=nm['name'], group=nm['group'],
                         voxels=int(sel.sum()), volume_mm3=float(sel.sum() * voxel_mm3), cells=float(c.sum()),
                         cells_fraction_of_tissue=float(c.sum() / cells_tissue),
                         density_cells_per_mm3=float(c.mean() / voxel_mm3),
                         cells_in_vermis=float(c[rm[sel] == 1].sum()),
                         cells_D99_left=float(c[xs < 0].sum()), cells_D99_right=float(c[xs >= 0].sum()),
                         cells_filled_rim=float(c[filled[..., meas][sel]].sum())))
        if has_cp:
            cc, nv = float(cpm[sel].sum()), float(arm[sel].sum() * voxel_mm3)
            rows[-1].update(cells_cp=cc, native_volume_mm3=nv, density_cp_cells_per_mm3=cc / nv if nv > 0 else float('nan'))
    cols = list(rows[0])
    with open(pre + '_labels.tsv', 'w') as fh:
        fh.write(f'# {cmd}  ({time.strftime("%Y-%m-%d %H:%M")}); transforms: {used}; measured sections only '
                 f'(interpolated sections {missing} left out); cells = sum of stack counts'
                 + ('; cells_cp, native_volume_mm3, density_cp: count-preserving stack (count_preserving_stack.py)'
                    if has_cp else '; no count-preserving stack (run count_preserving_stack.py for cells_cp)') + '\n')
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in cols) + '\n')

    # ---------------------------------------------------------------- QC
    summ = {'transforms': used, 'anterior lobe (I-V) vermis half-width (mm)': a.anterior_vermis_halfwidth_mm,
            'rim fill distance (mm)': a.fill_mm}
    for k, nm in REGION_NAMES.items():
        summ[f'cells: {nm} (fraction of tissue cells)'] = float(cm[rm == k].sum() / cells_tissue)
    summ['cells: filled rim (fraction of tissue cells, included above)'] = float(cm[filled[..., meas]].sum() / cells_tissue)
    summ['count-preserving columns (cells_cp, native_volume_mm3, density_cp)'] = 'yes' if has_cp else 'no (stack not found)'
    if has_cp:
        cp_tissue = float(cpm[lm > 0].sum())
        for k, nm in REGION_NAMES.items():
            summ[f'cells, count-preserving: {nm} (fraction of tissue cells)'] = float(cpm[rm == k].sum() / cp_tissue)
    per = {}
    for r in rows:
        if r['value'] in ANTERIOR_LOBE + VERMIS_POSTERIOR + HEMISPHERE + NUCLEI:
            per[f'{r["abbreviation"]}: {r["name"]}'] = (f'{r["cells"]:.3g} cells ({100 * r["cells_fraction_of_tissue"]:.1f}%), '
                                                        f'{r["density_cells_per_mm3"]:.0f} cells/mm^3, '
                                                        f'vermis {100 * r["cells_in_vermis"] / max(r["cells"], 1):.0f}%')
    ts = np.flatnonzero(meas & (lab > 0).any((0, 1)))
    slices = [int(s) for s in ts[np.linspace(0, len(ts) - 1, 8).round().astype(int)]] if len(ts) else []
    label_figure(counts, lab, reg, res_mm, slices, names, os.path.join(qc_dir, 'atlas_labels_sections.png'))
    across_sections_figure(counts, reg, res_mm, os.path.join(qc_dir, 'atlas_labels_across.png'))
    qc = dict(command=cmd, time=time.strftime('%Y-%m-%d %H:%M'),
              notes=['WORK IN PROGRESS: lobule labels not validated (LOBULE_LABELS_HANDOFF.md).',
                     f'D99 labels in the stack grid: {os.path.basename(pre)}_labels.nii.gz, _regions.nii.gz, _x_mm.nii.gz, '
                     f'_labels.tsv (see atlas_labels.py). D99 is a different animal; lobule boundaries can be off by '
                     f'about one folium. Lobules I-V are split into vermis / hemisphere at |D99 x| = '
                     f'{a.anterior_vermis_halfwidth_mm} mm (D99 has no vermis label there).'],
              tables={'D99 labels: summary': summ, 'D99 labels: cells per cerebellar label': per},
              images=[dict(file='atlas_labels_sections.png', caption='D99 labels on sagittal sections of the stack'),
                      dict(file='atlas_labels_across.png', caption='vermis / hemisphere / nuclei across the sections '
                                                                   '(the vermis is the band near the midline)')])
    with open(os.path.join(qc_dir, 'atlas_labels.json'), 'w') as fh:
        json.dump(qc, fh, indent=1, default=float)
    if not a.no_qc_report:
        subprocess.run([sys.executable, os.path.join(HERE, 'qc_report.py'), '--run-dir', run_dir,
                        '--rescale', str(a.rescale), '--subject', a.subject, '--missing', a.missing], check=False)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == '__main__':
    main()
