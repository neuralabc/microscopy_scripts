"""Block steps along a registered stack: offsets between groups of sections (e.g. sections mounted at a common angle).

For each boundary s, the mean of sections s-k+1..s (moving) is fitted rigidly (ANTs, GC metric) to the mean of
sections s+1..s+k (fixed). Per boundary: RMS displacement of the tissue (mm) and the rotation (deg). A block of
sections at a common angle shows as a pair of large steps of opposite sign (REGISTRATION_NOTES_local.md 4.8).

usage: uv run python macaque_CB/block_steps.py --run-dir <run> --rescale 20 [--level <level>] [--k 4]
           [--workers 6] [--lookup-csv <all_TP_image_idxs_file_lookup.csv>] [--out <file.csv>]
With --lookup-csv the table also has the TP (from the image paths) and image name on each side of the boundary.
"""
import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import nibabel
from scipy import ndimage as ndi

import env_compat  # noqa: F401
import ants
from tissue_mask import tissue_threshold_for

SUBJECT = 'zefir'
DEFAULT_LEVEL = 'coreg12nl_win12_rigsyn_4_groupwise_iter9'


def _fit(job):
    s, a, b, thr, res = job
    fi = ants.from_numpy(np.ascontiguousarray(b), spacing=(res, res))
    mv = ants.from_numpy(np.ascontiguousarray(a), spacing=(res, res))
    r = ants.registration(fi, mv, 'Rigid', aff_metric='GC', aff_random_sampling_rate=1.0, aff_iterations=(200, 100, 50),
                          aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0))
    t = ants.read_transform(r['fwdtransforms'][0])
    p, fc = np.array(t.parameters), np.array(t.fixed_parameters)
    A, tr = p[:4].reshape(2, 2), p[4:6]
    for f in set(r['fwdtransforms'] + r['invtransforms']):
        if os.path.isfile(f) and os.path.dirname(f) == '/tmp':
            os.remove(f)
    ii, jj = np.nonzero(ndi.binary_fill_holes(b > thr))
    pts = np.c_[ii, jj] * res
    disp = np.linalg.norm((pts - fc) @ A.T + tr + fc - pts, axis=1)
    return dict(boundary=s, rms_mm=float(np.sqrt((disp ** 2).mean())) if len(pts) else np.nan,
                rot_deg=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), tx_mm=float(tr[0]), ty_mm=float(tr[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--level', default=DEFAULT_LEVEL)
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--k', type=int, default=4, help='sections per side')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--lookup-csv', default=None)
    p.add_argument('--out', default=None, help='csv (default: <run>/QC/block_steps_<level>_k<k>.csv)')
    a = p.parse_args()
    res = a.rescale * 10 / 1000.0
    thr = tissue_threshold_for(res)
    S = np.asarray(nibabel.load(os.path.join(a.run_dir, f'{a.subject}_{a.level}_stack.nii.gz')).dataobj, np.float32)
    k, n = a.k, S.shape[2]
    jobs = [(s, S[..., s - k + 1:s + 1].mean(2), S[..., s + 1:s + k + 1].mean(2), thr, res) for s in range(k - 1, n - k)]
    with ProcessPoolExecutor(a.workers) as ex:
        d = pd.DataFrame(list(ex.map(_fit, jobs, chunksize=4)))
    if a.lookup_csv:
        df = pd.read_csv(a.lookup_csv)
        nat = sorted(glob.glob(os.path.join(a.run_dir, f'{a.subject}_[0-9][0-9][0-9][0-9]_*pix.nii.gz')))
        tp = [re.search(r'/(TP\d)/', f).group(1) if re.search(r'/(TP\d)/', f) else '' for f in df['file_name']]
        img = [os.path.basename(f)[len(a.subject) + 6:].replace('.nii.gz', '') for f in nat]
        for side, off in (('left', 0), ('right', 1)):
            d[f'tp_{side}'] = [tp[s + off] if s + off < len(tp) else '' for s in d.boundary]
            d[f'image_{side}'] = [img[s + off] if s + off < len(img) else '' for s in d.boundary]
    out = a.out or os.path.join(a.run_dir, 'QC', f'block_steps_{a.level}_k{k}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    print(f'{out}\nmedian RMS {d.rms_mm.median():.3f} mm, 95th pct {d.rms_mm.quantile(0.95):.3f}; boundaries > 0.5 mm: '
          f'{int((d.rms_mm > 0.5).sum())}; |rotation| > 3 deg: {int((d.rot_deg.abs() > 3).sum())}')
    print(d.sort_values('rms_mm', ascending=False).head(10).to_string(index=False, float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    main()
