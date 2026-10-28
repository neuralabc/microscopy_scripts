"""Remove the intermediate files of a slice-registration run and keep what is needed to push data (e.g. Mat's
Purkinje-cell maps) from the native sections into the final space.

Kept (never removed):
  - the native sections `<subject>_NNNN_<image>_pix.nii.gz` (the source grid of the first map)
  - the forward maps (`_ants-map.nii.gz`) of the transform chain to the final level, and their inverse maps
    (`_ants-invmap.nii.gz`, to pull final-space data back to the sections; --drop-inverse removes them). The chain
    comes from `<subject>_forward_maps.csv` through slice_registration_functions.get_transform_chain_for_level,
    the function that the mapping code uses: coreg0nl -> final STAGE1 level (rigsyn_iter4_win12) ->
    groupwise_iter3 -> ... -> groupwise_iterN
  - the deformed sections (`_ants-def0`) of the final level (to check a push against them)
  - all files that are not per-section files: stacks, tissue mask, D99 labels, csv / json / log files, the script
    copy, global templates; and the folders QC/, mri_registration/, missing_slices/, *.ome.zarr
Removed (only these patterns; any other file is kept and listed as "unknown"):
  - forward / inverse maps of levels that are not in the chain (cascade_0, STAGE1 levels 0-3 and the win1 / win2 /
    plain coreg12nl levels, coreg1nl / coreg2nl, groupwise_iter0-2)
  - deformed sections and deformed extra channels (`_ants-def0`, `_ants-def1`, ..., `_ants-def10`) of all levels
    except the def0 of the final level
  - ANTs intermediate files (`*_syn_1Warp.nii.gz`, `*_syn_1InverseWarp.nii.gz`, `*GenericAffine.mat`,
    `*.nii.gz_transformed.nii.gz`)
  - per-section templates (`*_template.nii.gz` of a section, `groupwise_iterK_local_template_sliceNNNN.nii.gz`)
After a cleanup, the columns of the removed levels in the maps csv files point to files that no longer exist; the
csv files are not changed. The QC report can no longer measure the removed levels (its cache keeps the old values).

Safety: a dry run is the default (it prints the plan). --execute removes the files, but only if every section has
its native file and all chain maps; it writes <run>/cleanup_manifest.txt (summary + every removed path).
--verify N pushes N sections through the kept chain (sequential maps, as Mat's notebook does) and compares the
result with the final deformed section; it reads kept files only, so it proves the cleanup keeps what is needed.

usage: uv run python macaque_CB/cleanup_run.py --run-dir <run> [--final-level groupwise_iter9] [--verify 5]
           [--drop-inverse] [--execute]
"""
import argparse
import collections
import os
import re
import sys
import time

import numpy as np
import pandas as pd

SUBJECT = 'zefir'
KEEP_DIRS = ('QC', 'mri_registration', 'missing_slices')


def rel_key(path):
    """Path relative to its run folder: the basename, plus the init_translation_slice_NNNN folder if it is in one."""
    parent = os.path.basename(os.path.dirname(path))
    return os.path.join(parent, os.path.basename(path)) if parent.startswith('init_translation_slice_') else os.path.basename(path)


def chain_files(run_dir, subject, final_col, first_chained):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import env_compat  # noqa: F401
    from slice_registration_functions import get_transform_chain_for_level
    fwd = pd.read_csv(os.path.join(run_dir, f'{subject}_forward_maps.csv'))
    inv_f = os.path.join(run_dir, f'{subject}_inverse_maps.csv')
    inv = pd.read_csv(inv_f) if os.path.isfile(inv_f) else None
    if final_col is None:
        gw = sorted((c for c in fwd.columns if re.fullmatch(r'groupwise_iter\d+', c)), key=lambda c: int(c[14:]))
        final_col = gw[-1]
    chain = get_transform_chain_for_level(fwd, final_col, groupwise_first_chained_iter=first_chained)
    fwd_keys = {c: [rel_key(p) for p in fwd[c]] for c in chain}
    inv_keys = {c: [rel_key(p) for p in inv[c]] for c in chain if inv is not None and c in inv.columns}
    return final_col, chain, fwd, fwd_keys, inv_keys


def classify(run_dir, subject, keep_fwd, keep_inv, final_def0_suffix, drop_inverse):
    """Return {category: [paths]} for every file under run_dir (folders in KEEP_DIRS and *.ome.zarr not walked)."""
    cats = collections.defaultdict(list)
    sec = re.compile(rf'^{subject}_\d{{4}}_')
    for dp, dn, fn in os.walk(run_dir):
        rel_dir = os.path.relpath(dp, run_dir)
        if rel_dir == '.':
            for d in list(dn):
                if d in KEEP_DIRS or d.endswith('.ome.zarr') or not d.startswith('init_translation_slice_'):
                    cats['keep: folder ' + (d if d in KEEP_DIRS else ('*.ome.zarr' if d.endswith('.ome.zarr') else 'other'))].append(os.path.join(dp, d))
                    dn.remove(d)
        for f in fn:
            p = os.path.join(dp, f)
            key = f if rel_dir == '.' else os.path.join(rel_dir, f)
            if key in keep_fwd:
                cats['keep: chain forward map'].append(p)
            elif key in keep_inv:
                cats['remove: chain inverse map (--drop-inverse)' if drop_inverse else 'keep: chain inverse map'].append(p)
            elif not sec.match(f):
                if re.fullmatch(r'groupwise_iter\d+_local_template_slice\d{4}\.nii\.gz', f):
                    cats['remove: per-section groupwise template'].append(p)
                else:
                    cats['keep: run-level file (stack, csv, log, ...)'].append(p)
            elif re.search(r'_pix\.nii\.gz$', f):
                cats['keep: native section'].append(p)
            elif f.endswith(final_def0_suffix):
                cats['keep: final-level deformed section (def0)'].append(p)
            elif re.search(r'_ants-(inv)?map\.nii\.gz$', f):
                cats['remove: map of a level not in the chain'].append(p)
            elif re.search(r'_ants-def\d+\.nii\.gz$', f):
                cats['remove: deformed section of a non-final level'].append(p)
            elif re.search(r'_syn_\d(Inverse)?Warp\.nii\.gz$|GenericAffine\.mat$|\.nii\.gz_transformed\.nii\.gz$', f):
                cats['remove: ANTs intermediate file'].append(p)
            elif f.endswith('_template.nii.gz'):
                cats['remove: per-section template'].append(p)
            else:
                cats['keep: unknown per-section file'].append(p)
    return cats


def size_of(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p)


def verify(run_dir, fwd, chain, final_col, n, missing):
    import nibabel
    from slice_registration_functions import apply_coordinate_mapping_2d
    final_def0 = {i: os.path.join(run_dir, rel_key(p).replace('_ants-map.nii.gz', '_ants-def0.nii.gz'))
                  for i, p in zip(fwd.slice_idx, fwd[final_col])}
    idx = [i for i in fwd.slice_idx if i not in missing]
    rows = []
    for i in [idx[k] for k in np.linspace(0, len(idx) - 1, n).round().astype(int)]:
        r = fwd[fwd.slice_idx == i].iloc[0]
        src = nibabel.load(os.path.join(run_dir, f'{SUBJECT}_{int(i):04d}_{r.slice_name}.nii.gz'))
        img = src
        for c in chain:
            img = apply_coordinate_mapping_2d(img, nibabel.load(os.path.join(run_dir, rel_key(r[c]))))
        a = np.asarray(img.dataobj, np.float64).squeeze()
        b = np.asarray(nibabel.load(final_def0[i]).dataobj, np.float64).squeeze()
        m = (a > 0) | (b > 0)
        ncc = float(np.corrcoef(a[m], b[m])[0, 1])
        rows.append((int(i), ncc, float(a.sum() / max(b.sum(), 1e-9))))
        print(f'  section {int(i):4d}: NCC with final def0 {ncc:.4f}, count ratio {rows[-1][2]:.4f}')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--final-level', default=None, help='maps csv column of the final level (default: last groupwise)')
    p.add_argument('--first-chained-iter', type=int, default=3, help='first groupwise iteration in the chain')
    p.add_argument('--drop-inverse', action='store_true', help='also remove the inverse maps of the chain')
    p.add_argument('--verify', type=int, default=0, help='push N sections through the kept chain and compare')
    p.add_argument('--missing', default='32,59,120,160,189,228', help='interpolated sections (left out of --verify)')
    p.add_argument('--execute', action='store_true', help='remove the files (default: dry run)')
    a = p.parse_args()
    run_dir = os.path.abspath(a.run_dir)

    final_col, chain, fwd, fwd_keys, inv_keys = chain_files(run_dir, a.subject, a.final_level, a.first_chained_iter)
    print(f'run: {run_dir}\nchain to {final_col}: {" -> ".join(chain)}')
    keep_fwd = {k for v in fwd_keys.values() for k in v}
    keep_inv = {k for v in inv_keys.values() for k in v}
    # suffix of the final level's deformed sections, from its map name (e.g. ..._groupwise_iter9_ants-def0.nii.gz)
    lvl = re.search(r'_pix_(.+)_ants-map\.nii\.gz$', os.path.basename(fwd[final_col].iloc[0])).group(1)
    final_def0_suffix = f'_pix_{lvl}_ants-def0.nii.gz'

    # completeness: every section needs its native file and all chain forward maps
    absent = [k for k in sorted(keep_fwd) if not os.path.isfile(os.path.join(run_dir, k))]
    natives = [f'{a.subject}_{int(i):04d}_{n}.nii.gz' for i, n in zip(fwd.slice_idx, fwd.slice_name)]
    absent += [k for k in natives if not os.path.isfile(os.path.join(run_dir, k))]
    print(f'sections: {len(fwd)}; chain forward maps present: {len(keep_fwd) - len([k for k in absent if k in keep_fwd])}'
          f'/{len(keep_fwd)}; missing needed files: {len(absent)}' + (f' (first: {absent[0]})' if absent else ''))

    cats = classify(run_dir, a.subject, keep_fwd, keep_inv, final_def0_suffix, a.drop_inverse)
    total = {'keep': [0, 0], 'remove': [0, 0]}
    print(f'\n{"GB":>8} {"files":>7}  category')
    for c in sorted(cats, key=lambda c: (not c.startswith('remove'), c)):
        s = sum(size_of(x) for x in cats[c])
        n = sum(sum(len(fs) for _, _, fs in os.walk(x)) if os.path.isdir(x) else 1 for x in cats[c])
        total[c.split(':')[0]][0] += n
        total[c.split(':')[0]][1] += s
        print(f'{s / 1e9:8.3f} {n:7d}  {c}')
    for k, (n, s) in total.items():
        print(f'{s / 1e9:8.3f} {n:7d}  TOTAL {k}')

    if a.verify:
        print(f'\nverify: {a.verify} sections pushed through the kept chain ({len(chain)} maps), against {final_col} def0')
        missing = {int(v) for v in a.missing.split(',') if v.strip()}
        rows = verify(run_dir, fwd, chain, final_col, a.verify, missing)
        bad = [r for r in rows if r[1] < 0.95]
        print('verify: OK' if not bad else f'verify: LOW NCC for sections {[r[0] for r in bad]}')

    if not a.execute:
        print('\ndry run: nothing removed (add --execute to remove the files in the "remove" categories)')
        return
    if absent:
        sys.exit('not removing anything: needed files are missing (an incomplete run?)')
    removed = [x for c in cats if c.startswith('remove') for x in cats[c]]
    man = os.path.join(run_dir, 'cleanup_manifest.txt')
    with open(man, 'a') as fh:
        fh.write(f'# cleanup_run.py {" ".join(sys.argv[1:])} ({time.strftime("%Y-%m-%d %H:%M")})\n'
                 f'# chain kept: {" -> ".join(chain)}; inverse maps {"removed" if a.drop_inverse else "kept"}\n'
                 f'# removed {total["remove"][0]} files, {total["remove"][1] / 1e9:.2f} GB\n')
        for x in removed:
            fh.write(os.path.relpath(x, run_dir) + '\n')
    for x in removed:
        os.remove(x)
    for d in os.listdir(run_dir):   # empty init_translation_slice folders (none expected: they keep the coreg0nl maps)
        dp = os.path.join(run_dir, d)
        if d.startswith('init_translation_slice_') and os.path.isdir(dp) and not os.listdir(dp):
            os.rmdir(dp)
    print(f'\nremoved {len(removed)} files ({total["remove"][1] / 1e9:.2f} GB); list in {man}')


if __name__ == '__main__':
    main()
