"""Per-section cell count of a registered level against the native section (sum of counts): how much the warps
change the area of each section. Ratio 1 = no change; the interpolated sections are left out.

usage: uv run python macaque_CB/count_ratio.py --levels coreg0nl,<level>,... NAME=<run_dir> [NAME=<run_dir> ...]
"""
import argparse
import glob
import os

import numpy as np
import nibabel

DEFAULT_MISSING = [32, 59, 120, 160, 189, 228]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('runs', nargs='+', help='NAME=<run_dir>')
    p.add_argument('--levels', default='coreg0nl,coreg12nl_win12_rigsyn_4,coreg12nl_win12_rigsyn_4_groupwise_iter9')
    p.add_argument('--subject', default='zefir')
    p.add_argument('--missing', default=','.join(map(str, DEFAULT_MISSING)))
    a = p.parse_args()
    missing = {int(v) for v in a.missing.split(',') if v.strip()}
    for name, d in (r.split('=', 1) for r in a.runs):
        nat = sorted(glob.glob(os.path.join(d, f'{a.subject}_[0-9][0-9][0-9][0-9]_*pix.nii.gz')))
        ns = np.array([np.asarray(nibabel.load(f).dataobj, np.float64).sum() for f in nat])
        keep = np.array([i not in missing for i in range(len(ns))])
        for lv in a.levels.split(','):
            S = np.asarray(nibabel.load(os.path.join(d, f'{a.subject}_{lv}_stack.nii.gz')).dataobj, np.float64)
            r = (S.sum((0, 1))[:len(ns)] / ns)[keep]
            dev = np.abs(r - 1)
            print(f'{name:6s} {lv:45s} ratio median {np.median(r):.3f}  |dev| median {np.median(dev):.3f}  '
                  f'95th {np.percentile(dev, 95):.3f}  max {dev.max():.3f}')


if __name__ == '__main__':
    main()
