"""Count-preserving (Jacobian-modulated) stack of a registration level, for cell-count totals per section / region.

The stacks of the pipeline (`<subject>_<level>_stack.nii.gz`) pull the native counts through the warps without a
Jacobian: each final pixel holds the native count per native pixel at that point, i.e. a density (counts per native
pixel area). That is right for looking at the anatomy and for masks, but a warp that stretches a section adds counts
and one that compresses it removes counts (QC "Counts": up to +28% at the stack end).

This tool pushes each native section through the same chain of pull maps (sequential, as the pipeline and Mat's
notebook do; the composition helpers in slice_registration_functions are marked broken) and multiplies the result of
each step by |det J| of that map (J in source pixels per output pixel). The sum of each output section is then the
native total, up to the interpolation error and tissue that a warp moves off the canvas (both reported per section).

Outputs (new files; the existing stacks and csv files are not changed):
  <subject>_<level>_counts_stack.nii.gz       cell counts per final pixel (sum over a region = cells in the region)
  <subject>_<level>_native_area_stack.nii.gz  native pixels per final pixel (a native image of ones, same chain)
  <subject>_<level>_counts_per_section.tsv    per section: native / modulated / unmodulated totals and ratios
  <subject>_<level>_counts_stack.json         summary, units, the chain, the interpolated sections, 'usage' rules
  <subject>_<level>_counts_stack_README.md    how to use the stacks (the 'usage' rules with this run's numbers)
Density of a region R (cells per mm^3) = sum_R(counts) / (sum_R(native_area) * native pixel area mm^2 * thickness mm).
Density image: the pipeline stack / (native pixel area mm^2 * thickness mm); not counts / native_area per voxel
(native_area is near 0 at the edges). usage_notes() holds these rules; qc_report.py and methods_report.py show them.
The interpolated sections (--missing) are 0 in both stacks and are left out of all totals: they are not real counts.

usage: uv run python macaque_CB/count_preserving_stack.py --run-dir <run> [--final-level groupwise_iter9]
           [--workers 8] [--thickness-um 50]
       --notes-only: (re)write the 'usage' rules and the README of an existing count stack (no recomputation)
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import nibabel

SUBJECT = 'zefir'
DEFAULT_MISSING = [32, 59, 120, 160, 189, 228]


def rel_key(path):
    """Path relative to its run folder (same rule as cleanup_run.py)."""
    parent = os.path.basename(os.path.dirname(path))
    return os.path.join(parent, os.path.basename(path)) if parent.startswith('init_translation_slice_') else os.path.basename(path)


def usage_notes(subject, level, px_um, thickness_um, missing=()):
    """How to use the count-preserving stack, the native area stack and the pipeline stack, with the numbers of the
    run (json 'usage', the README beside the stacks, the QC page). Returns (factor cells/mm^3 per native-pixel
    count, list of rules)."""
    voxel_mm3 = (px_um / 1000) ** 2 * (thickness_um / 1000)
    f = 1.0 / voxel_mm3
    stem = f'{subject}_{level}'
    rules = [
        f'Cells in a region R: the sum over R of {stem}_counts_stack.nii.gz (cells per final voxel).',
        f'Density of a region R (cells/mm^3) = sum_R(counts) / (sum_R(native_area) * {voxel_mm3:g} mm^3) = '
        f'{f:g} * sum_R(counts) / sum_R(native_area). native_area is {stem}_native_area_stack.nii.gz (native '
        f'pixels per final voxel); {voxel_mm3:g} mm^3 is one native pixel ({px_um:g} x {px_um:g} um) times the '
        f'section thickness ({thickness_um:g} um). Divide, do not multiply: the counts are not multiplied by the area.',
        f'Density image (cells/mm^3 per voxel): {f:g} * {stem}_stack.nii.gz. The pipeline stack already holds the '
        f'cells per native pixel at each point. Do not divide the counts stack by the native area voxel by voxel: '
        f'the native area is near 0 at the tissue and canvas edges, so that ratio is noisy there.',
        f'Use the counts stack for totals and region densities. Use the pipeline stack for images, masks and '
        f'boundaries. Sums of the pipeline stack are not cell totals, because the warps change the area '
        f'(see the unmodulated ratios in {stem}_counts_per_section.tsv).',
    ]
    if missing:
        rules.append(f'The interpolated sections {", ".join(map(str, sorted(missing)))} are 0 in both count-preserving '
                     f'stacks and are left out of all totals (they are not real counts).')
    return f, rules


def write_notes(stem, summary):
    """Add the usage rules to the summary (json) and write <stem>_counts_stack_README.md beside the stacks."""
    u = summary['units']
    px_um = 1000 * u['native_pixel_area_mm2'] ** 0.5
    th_um = 1000 * u['section_thickness_mm']
    subject_level = os.path.basename(stem)
    level = summary['level']
    subject = subject_level[:-len(level) - 1] if subject_level.endswith('_' + level) else subject_level
    f, rules = usage_notes(subject, level, px_um, th_um, summary.get('interpolated_sections_left_out', ()))
    u['density_factor_cells_per_mm3'] = f
    summary['usage'] = rules
    with open(stem + '_counts_stack.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    with open(stem + '_counts_stack_README.md', 'w') as fh:
        fh.write(f'# Count-preserving stack of {subject_level}\n\n'
                 f'Written by `count_preserving_stack.py` ({summary.get("time", "")}). Run folder: '
                 f'`{summary.get("run_dir", "")}`.\n\n'
                 + ''.join(f'- {r}\n' for r in rules)
                 + f'\nChain of maps: {" -> ".join(summary.get("chain", []))}. Checks: see '
                   f'`{subject_level}_counts_stack.json` and the QC page (Counts).\n')


def pull_modulated(src, coords):
    """One pull step with a Jacobian factor. src: (H0, W0) per-pixel quantity; coords: (H, W, 2) source voxel
    coordinates of each output pixel. Returns the output and the share of it on pixels with det J < 0 (folding;
    det J = 0 is common where a map samples a clamped canvas edge, and carries no counts)."""
    from scipy.ndimage import map_coordinates
    x, y = coords[..., 0], coords[..., 1]
    dx_di, dx_dj = np.gradient(x)
    dy_di, dy_dj = np.gradient(y)
    det = dx_di * dy_dj - dx_dj * dy_di
    inside = (x >= 0) & (x <= src.shape[0] - 1) & (y >= 0) & (y <= src.shape[1] - 1)
    jac = np.where(inside, np.abs(det), 0.0)   # border gradients of out-of-canvas samples are not meaningful
    out = map_coordinates(src, [x, y], order=1, mode='constant', cval=0.0) * jac
    tot = out.sum()
    folded = float(out[inside & (det < 0)].sum() / tot) if tot > 0 else 0.0
    return out, folded


def push_section(native_file, map_files):
    counts = np.asarray(nibabel.load(native_file).dataobj, np.float64).squeeze()
    area = np.ones_like(counts)
    folded = 0.0
    for f in map_files:
        coords = np.asarray(nibabel.load(f).dataobj, np.float64)
        counts, fo = pull_modulated(counts, coords)
        area, _ = pull_modulated(area, coords)
        folded = max(folded, fo)
    native = np.asarray(nibabel.load(native_file).dataobj, np.float64)
    return counts.astype(np.float32), area.astype(np.float32), float(native.sum()), float(native.size), folded


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--final-level', default=None, help='maps csv column of the level (default: last groupwise)')
    p.add_argument('--first-chained-iter', type=int, default=3, help='first groupwise iteration in the chain')
    p.add_argument('--missing', default=','.join(map(str, DEFAULT_MISSING)), help='interpolated sections (set to 0)')
    p.add_argument('--thickness-um', type=float, default=50.0, help='section thickness (for the density note)')
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLICEREG_COUNTS_WORKERS', 8)))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--notes-only', action='store_true', help='only (re)write the usage rules and the README')
    a = p.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import env_compat  # noqa: F401
    from slice_registration_functions import get_transform_chain_for_level

    fwd = pd.read_csv(os.path.join(run_dir, f'{a.subject}_forward_maps.csv'))
    col = a.final_level
    if col is None:
        col = sorted((c for c in fwd.columns if re.fullmatch(r'groupwise_iter\d+', c)), key=lambda c: int(c[14:]))[-1]
    chain = get_transform_chain_for_level(fwd, col, groupwise_first_chained_iter=a.first_chained_iter)
    level = re.search(r'_pix_(.+)_ants-map\.nii\.gz$', os.path.basename(fwd[col].iloc[0])).group(1)
    stem = os.path.join(run_dir, f'{a.subject}_{level}')
    out_counts, out_area = stem + '_counts_stack.nii.gz', stem + '_native_area_stack.nii.gz'
    if a.notes_only:
        with open(stem + '_counts_stack.json') as fh:
            summary = json.load(fh)
        write_notes(stem, summary)
        print(f'written: {stem}_counts_stack_README.md (and the usage rules in {stem}_counts_stack.json)')
        return
    if os.path.isfile(out_counts) and not a.overwrite:
        sys.exit(f'{out_counts} exists (use --overwrite)')
    ref = nibabel.load(stem + '_stack.nii.gz')   # grid, affine and section order of the pipeline stack
    nz = ref.shape[2]
    missing = {int(v) for v in a.missing.split(',') if v.strip()}
    print(f'run: {run_dir}\nlevel: {level}\nchain: {" -> ".join(chain)}\nsections: {len(fwd)} '
          f'(stack {nz}), interpolated (left out): {sorted(missing)}; workers {a.workers}')

    jobs = {}
    for _, r in fwd.iterrows():
        i = int(r.slice_idx)
        if i in missing or i >= nz:
            continue
        jobs[i] = (os.path.join(run_dir, f'{a.subject}_{i:04d}_{r.slice_name}.nii.gz'),
                   [os.path.join(run_dir, rel_key(r[c])) for c in chain])
    counts = np.zeros(ref.shape, np.float32)
    area = np.zeros(ref.shape, np.float32)
    rows = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {i: ex.submit(push_section, *jobs[i]) for i in sorted(jobs)}
        for k, (i, f) in enumerate(futs.items()):
            c, ar, ntot, npix, fo = f.result()
            counts[..., i], area[..., i] = c, ar
            rows[i] = dict(native_total=ntot, native_pixels=npix, modulated_total=float(c.sum(dtype=np.float64)),
                           native_area_total=float(ar.sum(dtype=np.float64)), folded_fraction=fo)
            if (k + 1) % 50 == 0:
                print(f'  {k + 1}/{len(futs)} sections ({time.time() - t0:.0f} s)')

    unmod = np.asarray(ref.dataobj, np.float64).sum((0, 1))
    tab = pd.DataFrame([dict(section=i, **rows[i], unmodulated_total=float(unmod[i])) for i in sorted(rows)])
    tab['modulated_ratio'] = tab.modulated_total / tab.native_total
    tab['unmodulated_ratio'] = tab.unmodulated_total / tab.native_total
    tab['area_ratio'] = tab.native_area_total / tab.native_pixels
    tab.to_csv(stem + '_counts_per_section.tsv', sep='\t', index=False, float_format='%.6g')

    hdr = ref.header.copy()
    hdr.set_data_dtype(np.float32)
    nibabel.save(nibabel.Nifti1Image(counts, ref.affine, hdr), out_counts)
    nibabel.save(nibabel.Nifti1Image(area, ref.affine, hdr), out_area)

    dev_m, dev_u = (tab.modulated_ratio - 1).abs(), (tab.unmodulated_ratio - 1).abs()
    px_um = float(ref.header.get_zooms()[0])
    summary = {
        'level': level, 'chain': chain, 'run_dir': run_dir,
        'interpolated_sections_left_out': sorted(missing),
        'units': {'counts_stack': 'cells per final pixel (sum over a region = cells)',
                  'native_area_stack': 'native pixels per final pixel',
                  'native_pixel_area_mm2': (px_um / 1000) ** 2, 'section_thickness_mm': a.thickness_um / 1000,
                  'density': 'cells/mm^3 = sum(counts) / (sum(native_area) * native_pixel_area_mm2 * thickness_mm)'},
        'check': {'modulated_ratio_abs_dev_median': float(dev_m.median()),
                  'modulated_ratio_abs_dev_p95': float(dev_m.quantile(.95)),
                  'modulated_ratio_abs_dev_max': float(dev_m.max()),
                  'unmodulated_ratio_abs_dev_median': float(dev_u.median()),
                  'unmodulated_ratio_abs_dev_max': float(dev_u.max()),
                  'folded_count_fraction_max': float(tab.folded_fraction.max()),
                  'sections_modulated_dev_over_2pct': tab.section[dev_m > .02].astype(int).tolist()},
        'command': 'count_preserving_stack.py ' + ' '.join(sys.argv[1:]),
        'time': time.strftime('%Y-%m-%d %H:%M'),
    }
    write_notes(stem, summary)   # writes the json (with the usage rules) and the README
    c = summary['check']
    print(f'modulated / native:   |dev| median {c["modulated_ratio_abs_dev_median"]:.4f}, '
          f'95th {c["modulated_ratio_abs_dev_p95"]:.4f}, max {c["modulated_ratio_abs_dev_max"]:.4f}')
    print(f'unmodulated / native: |dev| median {c["unmodulated_ratio_abs_dev_median"]:.4f}, '
          f'max {c["unmodulated_ratio_abs_dev_max"]:.4f}')
    print(f'sections with modulated |dev| > 2%: {c["sections_modulated_dev_over_2pct"]}; '
          f'max share of counts on folded pixels {c["folded_count_fraction_max"]:.4f}')
    print(f'written: {out_counts}\n         {out_area}\n         {stem}_counts_per_section.tsv / _counts_stack.json '
          f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
