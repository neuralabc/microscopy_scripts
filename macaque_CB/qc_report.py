"""QC report of one slice-registration run: a self-contained HTML page in <run_dir>/QC/index.html.

Run automatically at the end of run_slice_registration_optimized_newReg_sdf_v2_local.py, or by hand:
  uv run python macaque_CB/qc_report.py --run-dir <output_dir> --rescale 5

Content (all from the run's own files; nothing in the run directory is changed outside QC/):
  - run settings (run_settings.json, written by the main script) and the levels found
  - continuity per level: adjacent-slice NCC, blur-equalized NCC (same in-plane smoothing for every level, because
    raw NCC rises with interpolation blur) and adjacent tissue-mask Dice; median and 5th percentile
  - per-slice continuity along the stack, and the weakest adjacent pairs as images
  - warp: STAGE1 final maps (coreg0nl -> rigsyn_4) per slice; groupwise maps per iteration
  - per-slice total counts / native counts: an independent check (the registration does not optimise it); a change
    means that the warp stretched or compressed the slice
  - if count_preserving_stack.py has run for the final level: its count ratio (should be 1), and the cell total of
    each section, split by the D99 regions (atlas_labels.py) when they exist
  - slow path (low-pass tissue centroid and principal-axis angle), jitter poses (if the jitter correction ran)
  - through-plane views and a slice montage
  - MRI registration results (QC/mri/<level>/qc.json, written by mri_register.py), if any
  - the methods of the run (methods_report.py): QC/METHODS.md, written from the run's records, and shown here

Stacks finer than 0.2 mm are block-summed in-plane to about 0.2 mm for the measures and images (stated in the page),
so the report stays fast and the numbers are comparable between rescales 5, 10 and 20. Per-level measures are cached
in QC/cache/ and recomputed only when the stack file is newer (or with --recompute).
"""
import argparse
import base64
import glob
import html
import io
import json
import os
import re
import time

import numpy as np
import nibabel
from scipy import ndimage as ndi

import env_compat  # noqa: F401  (threads, OOM priority)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from evaluate_registration import adjacent_ncc, blur_equalized_ncc, adjacent_mask_dice, slow_path  # noqa: E402
import mri_anchor  # noqa: E402

SUBJECT = 'zefir'
SLICE_MM = 0.05
DEFAULT_MISSING = [32, 59, 120, 160, 189, 228]
QC_GRID_MM = 0.2
# reference categorical palette (dataviz skill), fixed order
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GREY = '#8a8984'

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 1.5})


# ----------------------------------------------------------------------------------------------------------------
# levels and data
# ----------------------------------------------------------------------------------------------------------------
def find_levels(run_dir, subject):
    """Levels with a stack file, in pipeline order: coreg0nl, cascade_0, STAGE1 (win12) levels, groupwise."""
    have = {os.path.basename(f)[len(subject) + 1:-len('_stack.nii.gz')]
            for f in glob.glob(os.path.join(run_dir, f'{subject}_*_stack.nii.gz'))}
    out = [lv for lv in ('coreg0nl', 'cascade_0') if lv in have]
    s1 = sorted((lv for lv in have if re.fullmatch(r'coreg12nl_win12_rigsyn_\d+', lv)), key=lambda v: int(v.split('_')[-1]))
    out += s1
    if s1:
        # fullmatch: count_preserving_stack.py writes <level>_counts_stack / _native_area_stack, which are not levels
        gw = [lv for lv in have if re.fullmatch(re.escape(s1[-1]) + r'_groupwise_iter\d+', lv)]
        out += sorted(gw, key=lambda v: int(v.split('iter')[-1]))
    return out


def short(level):
    return (level.replace('coreg12nl_win12_', '').replace('rigsyn_4_groupwise_iter', 'gw')
            .replace('rigsyn_', 'S1-'))


def block_sum(a, f):
    if f == 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f, *a.shape[2:]).sum(axis=(1, 3))


def load_stack(run_dir, subject, level, f):
    img = nibabel.load(os.path.join(run_dir, f'{subject}_{level}_stack.nii.gz'))
    return block_sum(np.asarray(img.dataobj, dtype=np.float32), f)


def slice_index(fname, subject):
    m = re.search(rf'{subject}_(\d{{4}})_', os.path.basename(fname))
    return int(m.group(1)) if m else None


def native_sums(run_dir, subject):
    files = sorted(glob.glob(os.path.join(run_dir, f'{subject}_[0-9][0-9][0-9][0-9]_*pix.nii.gz')))
    out = {}
    for f in files:
        out[slice_index(f, subject)] = float(np.asarray(nibabel.load(f).dataobj, dtype=np.float64).sum())
    return out


def map_warp(run_dir, subject, level, res_mm, n, thr):
    """Per slice: RMS (mm) of the whole displacement and of its non-affine part inside the tissue of the deformed
    slice; NaN where a slice has no map at this level."""
    total, nonaff = np.full(n, np.nan), np.full(n, np.nan)
    for f in glob.glob(os.path.join(run_dir, f'{subject}_[0-9][0-9][0-9][0-9]_*_{level}_ants-map.nii.gz')):
        s = slice_index(f, subject)
        if s is None or s >= n:
            continue
        m = np.asarray(nibabel.load(f).dataobj, dtype=np.float64)
        d0 = f.replace('_ants-map.nii.gz', '_ants-def0.nii.gz')
        tissue = (np.asarray(nibabel.load(d0).dataobj) > thr) if os.path.isfile(d0) else np.ones(m.shape[:2], bool)
        ii, jj = np.nonzero(tissue)
        if len(ii) < 10:
            continue
        if len(ii) > 20000:   # subsample big slices, the RMS does not need all voxels
            k = np.random.default_rng(0).choice(len(ii), 20000, replace=False)
            ii, jj = ii[k], jj[k]
        src = m[ii, jj, :2]
        grid = np.stack([ii, jj, np.ones_like(ii)], axis=1).astype(float)
        coef, *_ = np.linalg.lstsq(grid, src, rcond=None)
        nonaff[s] = np.sqrt(((src - grid @ coef) ** 2).sum(axis=1).mean()) * res_mm
        total[s] = np.sqrt(((src - grid[:, :2]) ** 2).sum(axis=1).mean()) * res_mm
    return total, nonaff


def level_measures(run_dir, subject, level, f, thr_qc, res_full, cache_dir, recompute):
    stack_f = os.path.join(run_dir, f'{subject}_{level}_stack.nii.gz')
    cache_f = os.path.join(cache_dir, f'{level}.npz')
    if not recompute and os.path.isfile(cache_f) and os.path.getmtime(cache_f) > os.path.getmtime(stack_f):
        return dict(np.load(cache_f))
    S = load_stack(run_dir, subject, level, f)
    out = dict(ncc=adjacent_ncc(S, thr_qc), beq=blur_equalized_ncc(S, thr_qc), dice=adjacent_mask_dice(S, thr_qc),
               sums=S.sum(axis=(0, 1)).astype(np.float64), path=slow_path(S, thr_qc, res_full * f))
    del S
    tissue_thr_full = mri_anchor.tissue_threshold_for(res_full)
    if level != 'coreg0nl':
        out['warp_total'], out['warp_nonaff'] = map_warp(run_dir, subject, level, res_full,
                                                         len(out['sums']), tissue_thr_full)
    np.savez(cache_f, **out)
    return out


# ----------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------
def fig_png(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def mark_missing(ax, missing):
    for s in missing:
        ax.axvline(s, color=GREY, lw=0.6, ls=':', zorder=0)


def fig_levels(levels, M):
    x = np.arange(len(levels))
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    for k, (key, lab) in enumerate([('beq', 'blur-equalized adjacent NCC'), ('dice', 'adjacent tissue-mask Dice')]):
        med = [np.median(M[lv][key]) for lv in levels]
        p5 = [np.percentile(M[lv][key], 5) for lv in levels]
        ax[k].plot(x, med, '-o', ms=4, color=C1, label='median')
        ax[k].plot(x, p5, '-o', ms=4, color=C2, label='5th percentile (weakest pairs)')
        ax[k].set_xticks(x, [short(lv) for lv in levels], rotation=60, fontsize=8)
        ax[k].set_title(lab)
        ax[k].legend(loc='lower right')
    return fig_png(fig)


def fig_continuity(levels_show, M, missing, worst):
    fig, ax = plt.subplots(len(levels_show), 1, figsize=(13, 2.2 * len(levels_show)), sharex=True, squeeze=False)
    for k, lv in enumerate(levels_show):
        a = ax[k, 0]
        y = M[lv]['beq']
        a.plot(np.arange(len(y)) + 0.5, y, color=C1, lw=1)
        mark_missing(a, missing)
        a.set_ylabel('blur-eq NCC')
        a.set_title(f'{lv}: adjacent pairs (pair s / s+1 at x = s + 0.5); dotted: interpolated slices', loc='left')
        if k == len(levels_show) - 1:
            for s in worst:
                a.plot(s + 0.5, y[s], 'o', ms=6, mfc='none', mec=C2, mew=1.5)
    ax[-1, 0].set_xlabel('slice')
    return fig_png(fig)


def fig_warp(stage1, gw_levels, M, missing):
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw=dict(width_ratios=[3, 1]))
    if stage1 is not None and 'warp_total' in M[stage1]:
        n = len(M[stage1]['warp_total'])
        ax[0].plot(np.arange(n), M[stage1]['warp_total'], color=C1, lw=1, label='total (rigid + affine + non-affine)')
        ax[0].plot(np.arange(n), M[stage1]['warp_nonaff'], color=C2, lw=1, label='non-affine part')
        mark_missing(ax[0], missing)
        ax[0].set_title(f'STAGE1 map {stage1} (from coreg0nl): RMS displacement in tissue', loc='left')
        ax[0].set_xlabel('slice'); ax[0].set_ylabel('mm'); ax[0].legend(loc='upper right')
    it = [(int(lv.split('iter')[-1]), M[lv]) for lv in gw_levels if 'warp_total' in M[lv]
          and np.isfinite(M[lv]['warp_total']).any()]
    if it:
        ax[1].plot([i for i, _ in it], [np.nanmedian(m['warp_total']) for _, m in it], '-o', ms=4, color=C1, label='total')
        ax[1].plot([i for i, _ in it], [np.nanmedian(m['warp_nonaff']) for _, m in it], '-o', ms=4, color=C2,
                   label='non-affine')
        ax[1].set_title('groupwise step warp (median)', loc='left')
        ax[1].set_xlabel('groupwise iteration'); ax[1].set_ylabel('mm'); ax[1].legend()
    else:
        ax[1].axis('off')
    return fig_png(fig)


def fig_counts(levels_show, ratios, missing):
    fig, ax = plt.subplots(1, 1, figsize=(13, 3.2))
    for lv, col in zip(levels_show, (C1, C2, C3, C4)):
        r = ratios[lv]
        ax.plot(np.arange(len(r)), r, color=col, lw=1, label=lv)
    ax.axhspan(0.95, 1.05, color='#e4e3df', alpha=0.6, zorder=0, lw=0)
    mark_missing(ax, missing)
    ax.set_title('per-slice total counts / native counts (band: +/- 5%; 1 = no area change)', loc='left')
    ax.set_xlabel('slice'); ax.set_ylabel('ratio'); ax.legend(loc='lower right')
    return fig_png(fig)


def count_preserving(run_dir, subject, final, missing):
    """Per-section totals of the count-preserving stack (count_preserving_stack.py) of the final level, split by the
    D99 regions of atlas_labels.py when present. Returns None if the tool has not run."""
    import pandas as pd
    pre = os.path.join(run_dir, f'{subject}_{final}_')
    tf = pre + 'counts_per_section.tsv'
    if not os.path.isfile(tf):
        return None
    tab = pd.read_csv(tf, sep='\t')
    groups = None
    rf, cf = pre + 'D99_regions.nii.gz', pre + 'counts_stack.nii.gz'
    if os.path.isfile(rf) and os.path.isfile(cf):
        reg = np.asarray(nibabel.load(rf).dataobj)
        cnt = np.asarray(nibabel.load(cf).dataobj, np.float64)
        # atlas_labels.py codes: 1 vermis, 2 hemisphere, 3 nuclei, 4 other D99 label, 252 cerebellum no label,
        # 253 brainstem no label, 254 neither, 255 tissue outside D99, 0 outside the tissue mask
        defs = [('cerebellum (D99 1-3, 252)', (1, 2, 3, 252)), ('brainstem / other D99 (4, 253)', (4, 253)),
                ('tissue outside D99 (254, 255)', (254, 255)), ('outside the tissue mask', (0,))]
        groups = {nm: np.array([cnt[..., k][np.isin(reg[..., k], v)].sum() for k in range(cnt.shape[2])])
                  for nm, v in defs}
        for g in groups.values():
            g[[m for m in missing if m < g.size]] = np.nan
        del reg, cnt
    return tab, groups, count_usage(pre + 'counts_stack.json', subject, final, missing)


def count_usage(json_file, subject, final, missing):
    """Usage rules of the count stacks (count_preserving_stack.usage_notes) with the numbers of the run: from the json
    when it has them, else from its units (runs before the rules were added). [] if there is no json."""
    if not os.path.isfile(json_file):
        return []
    with open(json_file) as fh:
        c = json.load(fh)
    if c.get('usage'):
        return c['usage']
    from count_preserving_stack import usage_notes
    u = c['units']
    return usage_notes(subject, final, 1000 * u['native_pixel_area_mm2'] ** 0.5, 1000 * u['section_thickness_mm'],
                       c.get('interpolated_sections_left_out', missing))[1]


def fig_totals(tab, groups, missing):
    fig, ax = plt.subplots(1, 1, figsize=(13, 3.4))
    ax.plot(tab.section, tab.native_total / 1e3, color=GREY, lw=1.2, label='native section (all)')
    if groups:
        for (nm, g), col in zip(groups.items(), (C1, C2, C3, C4)):
            ax.plot(np.arange(g.size), g / 1e3, color=col, lw=1, label=nm)
    mark_missing(ax, missing)
    ax.set_title('cells per section (thousands); count-preserving stack split by D99 region (WIP labels)', loc='left')
    ax.set_xlabel('slice'); ax.set_ylabel('cells (thousands)'); ax.legend(loc='upper left', fontsize=8)
    return fig_png(fig)


def fig_slow_path(levels_show, M, res_qc):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    for lv, col in zip(levels_show, (C1, C2, C3)):
        p = M[lv]['path']
        for k, lab in enumerate(['centroid i (mm)', 'centroid j (mm)', 'principal axis (deg)']):
            ax[k].plot(p[:, k] - (np.nanmean(p[:, k]) if k < 2 else 0), color=col, label=lv)
            ax[k].set_title(lab + (' - mean' if k < 2 else ''), loc='left'); ax[k].set_xlabel('slice')
    ax[0].legend(fontsize=7)
    return fig_png(fig)


def fig_jitter(poses_csv, missing):
    import pandas as pd
    d = pd.read_csv(poses_csv)
    fig, ax = plt.subplots(1, 2, figsize=(13, 2.8))
    ax[0].plot(d['slice_idx'], d['theta_deg'], color=C1, lw=1)
    ax[0].set_title('jitter correction: rotation (deg)', loc='left')
    ax[1].plot(d['slice_idx'], d['tx_mm'], color=C1, lw=1, label='tx')
    ax[1].plot(d['slice_idx'], d['ty_mm'], color=C2, lw=1, label='ty')
    ax[1].set_title('jitter correction: shift (mm)', loc='left'); ax[1].legend()
    for a in ax:
        mark_missing(a, missing); a.set_xlabel('slice')
    return fig_png(fig)


def _vmax(S):
    v = S[S > 0]
    return float(np.percentile(v, 99)) if v.size else 1.0


def tissue_box(S, thr, pad=3):
    """In-plane bounding box (slices) of the tissue in any slice of the stack."""
    m = (S > thr).any(axis=2)
    if not m.any():
        return slice(None), slice(None)
    ii, jj = np.nonzero(m)
    return (slice(max(ii.min() - pad, 0), ii.max() + pad + 1), slice(max(jj.min() - pad, 0), jj.max() + pad + 1))


def fig_through_plane(run_dir, subject, levels_show, f, res_qc, centre, box):
    ci, cj = centre
    bi, bj = box
    rows = []
    for lv in levels_show:
        S = load_stack(run_dir, subject, lv, f)
        rows.append((lv, S[ci, bj, :], S[bi, cj, :], S[bi, bj, S.shape[2] // 2], _vmax(S), S.shape[2]))
        del S
    n = rows[0][-1]
    ext_s = n * SLICE_MM
    hi, hj = rows[0][2].shape[0] * res_qc, rows[0][1].shape[0] * res_qc       # physical heights of the two cuts
    widths = [ext_s, ext_s, hj]
    fig, ax = plt.subplots(len(rows), 3, figsize=(15, 15 * max(hi, hj) / sum(widths) * len(rows) + 0.8 * len(rows)),
                           squeeze=False, gridspec_kw=dict(width_ratios=widths))
    for r, (lv, cut_i, cut_j, mid, vmax, n) in enumerate(rows):
        ax[r, 0].imshow(cut_i, cmap='gray', vmin=0, vmax=vmax, extent=(0, ext_s, cut_i.shape[0] * res_qc, 0))
        ax[r, 0].set_title(f'{short(lv)}: cut i = {ci}', loc='left')
        ax[r, 1].imshow(cut_j, cmap='gray', vmin=0, vmax=vmax, extent=(0, ext_s, cut_j.shape[0] * res_qc, 0))
        ax[r, 1].set_title(f'{short(lv)}: cut j = {cj}', loc='left')
        ax[r, 2].imshow(mid, cmap='gray', vmin=0, vmax=vmax, extent=(0, mid.shape[1] * res_qc, mid.shape[0] * res_qc, 0))
        ax[r, 2].set_title(f'{short(lv)}: slice {n // 2}', loc='left')
        for a in ax[r]:
            a.grid(False)
    ax[-1, 0].set_xlabel('along the stack (mm)'); ax[-1, 1].set_xlabel('along the stack (mm)')
    ax[-1, 2].set_xlabel('mm')
    fig.tight_layout()
    return fig_png(fig, dpi=90)


def fig_montage(S, box, n_show=24):
    S = S[box[0], box[1]]
    idx = np.linspace(0, S.shape[2] - 1, n_show).round().astype(int)
    cols = 8
    rows = int(np.ceil(n_show / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2 * cols, 1.6 * rows))
    vmax = _vmax(S)
    for a, s in zip(ax.ravel(), idx):
        a.imshow(S[..., s], cmap='gray', vmin=0, vmax=vmax); a.set_title(str(s), fontsize=8); a.axis('off')
    for a in ax.ravel()[len(idx):]:
        a.axis('off')
    return fig_png(fig)


def fig_worst_pairs(S, worst, box):
    S = S[box[0], box[1]]
    fig, ax = plt.subplots(len(worst), 3, figsize=(9, 2.6 * len(worst)), squeeze=False)
    vmax = _vmax(S)
    for r, s in enumerate(worst):
        a, b = S[..., s], S[..., s + 1]
        ax[r, 0].imshow(a, cmap='gray', vmin=0, vmax=vmax); ax[r, 0].set_title(f'slice {s}', fontsize=8)
        ax[r, 1].imshow(b, cmap='gray', vmin=0, vmax=vmax); ax[r, 1].set_title(f'slice {s + 1}', fontsize=8)
        rgb = np.stack([np.clip(a / vmax, 0, 1), np.clip(b / vmax, 0, 1), np.clip(a / vmax, 0, 1)], axis=2)
        ax[r, 2].imshow(rgb); ax[r, 2].set_title(f'overlay: {s} magenta, {s + 1} green', fontsize=8)
        for x in ax[r]:
            x.axis('off')
    return fig_png(fig)


# ----------------------------------------------------------------------------------------------------------------
# html
# ----------------------------------------------------------------------------------------------------------------
CSS = """
:root { --bg:#fcfcfb; --card:#ffffff; --text:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6;
        --bad:#b42318; --ok:#067647; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1a1a19; --card:#232322; --text:#f0efec;
        --muted:#c3c2b7; --line:#3a3a38; --accent:#3987e5; --bad:#f97066; --ok:#47cd89; color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#1a1a19; --card:#232322; --text:#f0efec; --muted:#c3c2b7; --line:#3a3a38;
        --accent:#3987e5; --bad:#f97066; --ok:#47cd89; color-scheme: dark; }
body { background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; margin:0; padding:0 16px 48px; }
main { max-width:1200px; margin:0 auto; }
h1 { font-size:22px; margin:24px 0 4px; } h2 { font-size:17px; margin:32px 0 8px; border-bottom:1px solid var(--line);
     padding-bottom:4px; } h3 { font-size:15px; margin:20px 0 6px; }
p, li { color:var(--muted); max-width:90ch; } code { font-size:12px; }
nav a { margin-right:14px; color:var(--accent); text-decoration:none; }
figure { margin:12px 0 20px; background:#fff; border:1px solid var(--line); border-radius:6px; padding:8px; }
figure img { max-width:100%; height:auto; display:block; } figcaption { color:#52514e; font-size:12px; margin-top:6px; }
.tablewrap { overflow-x:auto; } table { border-collapse:collapse; font-size:12.5px; margin:8px 0 16px; }
th, td { border-bottom:1px solid var(--line); padding:4px 10px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--muted); font-weight:600; }
.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
"""


def table(rows, header):
    h = ''.join(f'<th>{html.escape(str(c))}</th>' for c in header)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(b64, caption):
    return (f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return f'{v:.{nd}f}' if np.isfinite(v) else '-'
    return html.escape(str(v))


def mri_sections(qc_dir):
    out = []
    for qf in sorted(glob.glob(os.path.join(qc_dir, 'mri', '*', 'qc.json'))):
        d = os.path.dirname(qf)
        with open(qf) as fh:
            q = json.load(fh)
        status = q.get('status', '?')
        cls = 'ok' if status == 'ok' else 'bad'
        parts = [f'<h3>{html.escape(q.get("level", os.path.basename(d)))}</h3>',
                 f'<p>Status: <span class="{cls}">{html.escape(status)}</span>. Written {html.escape(q.get("time", ""))} '
                 f'by <code>{html.escape(q.get("command", "mri_register.py"))}</code></p>']
        if q.get('notes'):
            parts.append('<ul>' + ''.join(f'<li>{html.escape(n)}</li>' for n in q['notes']) + '</ul>')
        for title, tab in q.get('tables', {}).items():
            parts.append(f'<p><b>{html.escape(title)}</b></p>' + table([[html.escape(k), fmt(v)] for k, v in tab.items()],
                                                                      ['measure', 'value']))
        for im in q.get('images', []):
            p = os.path.join(d, im['file'])
            if os.path.isfile(p):
                with open(p, 'rb') as fh:
                    parts.append(figure(base64.b64encode(fh.read()).decode('ascii'), im.get('caption', im['file'])))
        lf = os.path.join(d, 'atlas_labels.json')   # written by atlas_labels.py
        if os.path.isfile(lf):
            with open(lf) as fh:
                lq = json.load(fh)
            parts.append(f'<h4>D99 labels in the stack</h4><p>Written {html.escape(lq.get("time", ""))} by '
                         f'<code>{html.escape(lq.get("command", "atlas_labels.py"))}</code></p>')
            if lq.get('notes'):
                parts.append('<ul>' + ''.join(f'<li>{html.escape(n)}</li>' for n in lq['notes']) + '</ul>')
            for title, tab in lq.get('tables', {}).items():
                parts.append(f'<p><b>{html.escape(title)}</b></p>' + table([[html.escape(k), fmt(v)] for k, v in tab.items()],
                                                                          ['measure', 'value']))
            for im in lq.get('images', []):
                p = os.path.join(d, im['file'])
                if os.path.isfile(p):
                    with open(p, 'rb') as fh:
                        parts.append(figure(base64.b64encode(fh.read()).decode('ascii'), im.get('caption', im['file'])))
        out.append('\n'.join(parts))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--missing', default=','.join(map(str, DEFAULT_MISSING)),
                   help='interpolated slice indices (comma separated, empty for none)')
    p.add_argument('--out', default=None, help='QC directory (default: <run_dir>/QC)')
    p.add_argument('--recompute', action='store_true', help='ignore the cached per-level measures')
    p.add_argument('--n-worst', type=int, default=6)
    p.add_argument('--qc-grid-mm', type=float, default=QC_GRID_MM,
                   help='in-plane grid of the QC measures and images (block sum of the stack)')
    a = p.parse_args()

    run_dir = a.run_dir.rstrip('/') + '/'
    qc_dir = a.out or os.path.join(run_dir, 'QC')
    cache_dir = os.path.join(qc_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    missing = [int(v) for v in a.missing.split(',') if v.strip()]
    res_full = a.rescale * 10 / 1000.0
    f = max(1, int(round(a.qc_grid_mm / res_full)))
    res_qc = res_full * f
    thr_qc = mri_anchor.tissue_threshold_for(res_qc)

    levels = find_levels(run_dir, a.subject)
    if not levels:
        raise SystemExit(f'no stacks found in {run_dir}')
    t0 = time.time()
    M = {}
    for lv in levels:
        M[lv] = level_measures(run_dir, a.subject, lv, f, thr_qc, res_full, cache_dir, a.recompute)
        print(f'QC: {lv} ({time.time() - t0:.0f} s)', flush=True)
    s1 = [lv for lv in levels if re.fullmatch(r'coreg12nl_win12_rigsyn_\d+', lv)]
    stage1 = s1[-1] if s1 else None
    gw = [lv for lv in levels if 'groupwise_iter' in lv]
    final = gw[-1] if gw else levels[-1]
    show = [lv for lv in ('coreg0nl', stage1, final) if lv is not None and lv in M]
    show = list(dict.fromkeys(show))

    nat = native_sums(run_dir, a.subject)
    n = len(M[final]['sums'])
    nat_arr = np.array([nat.get(s, np.nan) for s in range(n)])
    keep = np.array([s not in missing for s in range(n)])
    ratios = {}
    for lv in show:
        r = M[lv]['sums'][:n] / nat_arr
        r[~keep] = np.nan
        ratios[lv] = r

    beq_final = M[final]['beq'].copy()
    worst = [int(s) for s in np.argsort(beq_final)[:a.n_worst]]

    # summary table
    rows = []
    for lv in levels:
        m = M[lv]
        w = m.get('warp_total')
        rows.append([html.escape(lv)] + [fmt(v) for v in (
            np.median(m['ncc']), np.percentile(m['ncc'], 5), np.median(m['beq']), np.percentile(m['beq'], 5),
            np.median(m['dice']), np.percentile(m['dice'], 5))] +
            [fmt(np.nanmedian(w)) if w is not None and np.isfinite(w).any() else '-',
             fmt(np.nanmedian(m['warp_nonaff'])) if w is not None and np.isfinite(w).any() else '-'])
    level_table = table(rows, ['level', 'NCC median', 'NCC p5', 'blur-eq median', 'blur-eq p5', 'Dice median',
                               'Dice p5', 'map warp total (mm)', 'non-affine (mm)'])
    count_rows = []
    for lv in show:
        r = ratios[lv][np.isfinite(ratios[lv])]
        if r.size == 0:   # no native slices found
            count_rows.append([html.escape(lv)] + ['-'] * 5)
            continue
        dev = np.abs(r - 1)
        count_rows.append([html.escape(lv), fmt(np.median(r)), fmt(100 * np.median(dev), 1) + ' %',
                           fmt(100 * np.percentile(dev, 95), 1) + ' %', fmt(100 * dev.max(), 1) + ' %',
                           str(int(np.argmax(np.nan_to_num(np.abs(ratios[lv] - 1)))))])
    cp = count_preserving(run_dir, a.subject, final, missing)
    if cp is not None:   # same measure for the count-preserving stack of the final level
        rcp = np.full(n, np.nan)
        rcp[cp[0].section.values.astype(int)] = cp[0].modulated_ratio.values
        rcp[~keep] = np.nan
        ratios[final + ' (count-preserving)'] = rcp
        dev = np.abs(rcp[np.isfinite(rcp)] - 1)
        count_rows.append([html.escape(final + ' (count-preserving)'), fmt(np.nanmedian(rcp)),
                           fmt(100 * np.median(dev), 2) + ' %', fmt(100 * np.percentile(dev, 95), 2) + ' %',
                           fmt(100 * dev.max(), 2) + ' %', str(int(np.nanargmax(np.abs(rcp - 1))))])
    count_table = table(count_rows, ['level', 'ratio median', '|dev| median', '|dev| 95th', '|dev| max', 'slice of max'])

    # settings
    settings = {}
    sf = os.path.join(run_dir, 'run_settings.json')
    if os.path.isfile(sf):
        with open(sf) as fh:
            settings = json.load(fh)
    settings_rows = [[html.escape(k), html.escape(json.dumps(v) if not isinstance(v, str) else v)]
                     for k, v in settings.items()]
    settings_rows += [['QC grid', f'{res_qc:.3f} mm in-plane (block sum x{f} of {res_full:.3f} mm), '
                       f'{SLICE_MM} mm sections; tissue threshold {thr_qc:g} counts/voxel']]

    centre_stack = load_stack(run_dir, a.subject, final, f)
    proj = centre_stack.sum(axis=2)
    ci, cj = [int(round(v)) for v in ndi.center_of_mass(proj > np.percentile(proj, 80))]
    box = tissue_box(centre_stack, thr_qc)

    sections = []
    sections.append(('settings', 'Run', table(settings_rows, ['setting', 'value']) +
                     f'<p>Levels found: {len(levels)}. Final level: <code>{html.escape(final)}</code>. '
                     f'Interpolated slices: {html.escape(str(missing))}.</p>'))
    sections.append(('levels', 'Continuity per level',
                     '<p>Adjacent-slice similarity at each level. Blur-equalized NCC smooths every level the same way '
                     'in-plane before the NCC, so levels with more resampling steps do not look better only because they '
                     'are blurrier. The 5th percentile shows the weakest pairs. Groupwise iterations 0-2 are not chained '
                     'and do not change the stack.</p>' + figure(fig_levels(levels, M), 'continuity per level') + level_table))
    sections.append(('slices', 'Continuity along the stack',
                     figure(fig_continuity(show, M, missing, worst),
                            f'blur-equalized NCC of adjacent pairs; circles: the {a.n_worst} weakest pairs of the final level')
                     + figure(fig_worst_pairs(centre_stack, worst, box),
                              f'the {a.n_worst} weakest adjacent pairs of {final}')))
    sections.append(('warp', 'Warp', '<p>RMS displacement of the coordinate maps inside the tissue. The STAGE1 map '
                     'starts at coreg0nl; each groupwise map is one step.</p>' + figure(fig_warp(stage1, gw, M, missing),
                                                                                       'warp per slice and per groupwise step')))
    sections.append(('counts', 'Counts (independent check)',
                     '<p>Total cell counts of each registered slice divided by those of its native slice. The '
                     'registration does not optimise this. A ratio below 1 means that the warp compressed the slice '
                     '(or moved tissue out of the canvas); above 1, stretched. Interpolated slices are left out.</p>'
                     + figure(fig_counts(list(ratios), ratios, missing), 'count ratio per slice') + count_table
                     + ('<p>The count-preserving stack (<code>count_preserving_stack.py</code>) multiplies each pull '
                        'step by |det J| of its map, so that its section totals keep the native totals. Use it for '
                        'cell totals per region; use the pipeline stack (counts per native pixel area) for images and '
                        'masks.</p>'
                        + ('<p><b>How to use the stacks</b> (also in <code>' + html.escape(f'{a.subject}_{final}')
                           + '_counts_stack_README.md</code> beside the stacks):</p><ul>'
                           + ''.join(f'<li>{html.escape(r)}</li>' for r in cp[2]) + '</ul>' if cp[2] else '')
                        + figure(fig_totals(cp[0], cp[1], missing), 'cells per section')
                        if cp is not None else
                        '<p>No count-preserving stack for the final level (run <code>count_preserving_stack.py</code>).'
                        '</p>')))
    sections.append(('views', 'Through-plane views',
                     figure(fig_through_plane(run_dir, a.subject, show, f, res_qc, (ci, cj), box),
                            'through-plane cuts at the tissue centre (physical aspect) and the middle slice')
                     + figure(fig_montage(centre_stack, box), f'{final}: evenly spaced slices')))
    path_html = figure(fig_slow_path(show, M, res_qc), 'slow path (low-pass, sigma 15 slices)')
    jit_csv = os.path.join(run_dir, f'{a.subject}_coreg0nl_jitter_poses.csv')
    if os.path.isfile(jit_csv):
        path_html += figure(fig_jitter(jit_csv, missing), 'rigid jitter correction folded into coreg0nl')
    sections.append(('path', 'Slow path and jitter correction',
                     '<p>The slow path of coreg0nl follows the scan framing (no physical reference); compare the '
                     'registered levels with it to see how much the registration changed the overall shape.</p>' + path_html))
    mri = mri_sections(qc_dir)
    sections.append(('mri', 'MRI registration (D99)', '\n'.join(mri) if mri else
                     '<p>Not run for this output. Run <code>uv run python macaque_CB/mri_register.py --run-dir '
                     f'{html.escape(run_dir)} --rescale {a.rescale} --level &lt;level&gt;</code>; it adds its results '
                     'here.</p>'))
    del centre_stack
    try:   # methods text of this run (methods_report.py): QC/METHODS.md, also shown here
        import methods_report
        mpath, md, _checks = methods_report.write(run_dir, a.subject, missing)
        sections.append(('methods', 'Methods', '<p>Generated from the records of this run; the same text is in '
                         '<a href="METHODS.md"><code>QC/METHODS.md</code></a>.</p>' + methods_report.md_to_html(md)))
    except Exception as e:
        sections.append(('methods', 'Methods', f'<p>The methods text could not be written: {html.escape(str(e))}</p>'))

    nav = ''.join(f'<a href="#{k}">{html.escape(t)}</a>' for k, t, _ in sections)
    body = ''.join(f'<h2 id="{k}">{html.escape(t)}</h2>{c}' for k, t, c in sections)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Slice registration QC</title><style>{CSS}</style></head><body><main>'
            f'<h1>Slice registration QC</h1><p><code>{html.escape(run_dir)}</code><br>Written '
            f'{time.strftime("%Y-%m-%d %H:%M")} by qc_report.py</p><nav>{nav}</nav>{body}</main></body></html>')
    with open(os.path.join(qc_dir, 'index.html'), 'w') as fh:
        fh.write(page)
    summary = {lv: dict(ncc_median=float(np.median(M[lv]['ncc'])), ncc_p5=float(np.percentile(M[lv]['ncc'], 5)),
                        blur_eq_ncc_median=float(np.median(M[lv]['beq'])), blur_eq_ncc_p5=float(np.percentile(M[lv]['beq'], 5)),
                        mask_dice_median=float(np.median(M[lv]['dice'])), mask_dice_p5=float(np.percentile(M[lv]['dice'], 5)))
               for lv in levels}
    with open(os.path.join(qc_dir, 'metrics.json'), 'w') as fh:
        json.dump(dict(run_dir=run_dir, rescale=a.rescale, qc_grid_mm=res_qc, final_level=final, worst_pairs=worst,
                       levels=summary), fh, indent=1)
    print(f'QC report: {os.path.join(qc_dir, "index.html")} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
