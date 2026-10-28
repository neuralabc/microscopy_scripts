"""Rigid per-slice pose correction from a pose graph over several slice distances (no warping).

Why: the jitter correction (slice_jitter.py) fits each slice to its two nearest neighbours. A block of consecutive
sections that were mounted at a common angle (e.g. 276-283 of zefir, about 15 deg against both neighbours) is
consistent inside, so only its two edges show a mismatch and the nearest-neighbour fit cannot turn the whole block
back (it only smooths its edges). Here every slice is fitted rigidly to the slices at several distances
(default 1, 2, 3, 4, 6, 8, 12, 16), and one least-squares solve finds the poses that agree best with all pairs: a
pair that crosses a block edge sees the block angle at every distance, so the block is corrected as a unit.

Solve (poses as in slice_jitter: rotate about the slice's tissue centroid, then translate, in deg and mm):
  1. rotations: theta_s - theta_u = phi_su for each pair (phi_su: rotation that brings s onto u), weight 1/d,
     robust (Huber, by iterative reweighting), mean rotation fixed to 0;
  2. translations, with the rotations fixed: t_s - t_u = R(theta_u) (q_su - c_u) + c_u - c_s, where q_su is the
     place of s's centroid on u (from the same fit);
  3. the slow part of the poses (Gaussian, sigma slices) is removed, as in the jitter correction, so the slow path
     of the stack (where it sits as a whole and how it bends) is kept;
  4. repeated (outer iterations) on the corrected stack; the poses add up exactly because each pass rotates about
     the moved centroid.
The poses are folded into a level with slice_jitter.correct_level_in_place(..., poses_fn=posegraph_poses).
"""
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi

from slice_jitter import _rigid_to_target, _low_pass, apply_poses, adjacent_ncc

DEFAULT_DISTANCES = (1, 2, 3, 4, 6, 8, 12, 16)


def _fit_pair(job):
    s, u, mov, target, thr, res_mm = job
    r = _rigid_to_target((mov, target, thr, res_mm))
    return (s, u, r)


def pair_measurements(stack, valid, distances, thr, res_mm, max_workers=1):
    """Rigid fit of slice s (moving) onto slice s+d (fixed) for all valid pairs; returns [(s, u, phi_deg, dx_mm, dy_mm)]."""
    n = stack.shape[2]
    jobs = [(s, s + d, stack[..., s], stack[..., s + d], thr, res_mm)
            for d in distances for s in range(n - d) if valid[s] and valid[s + d]]
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            res = list(ex.map(_fit_pair, jobs, chunksize=8))
    else:
        res = [_fit_pair(j) for j in jobs]
    return [(s, u, *r) for s, u, r in res if r is not None]


def _huber_lsq(A, b, w, n_iter=10, k_mad=2.5):
    """Weighted least squares with Huber reweighting of the residuals."""
    wr = np.ones(len(b))
    for _ in range(n_iter):
        W = np.sqrt(w * wr)
        x, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        r = A @ x - b
        sc = 1.4826 * np.median(np.abs(r)) + 1e-9
        k = k_mad * sc
        wr = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return x, r, wr


def solve_poses(meas, centroids_mm, valid):
    """Poses (n x 3: theta_deg, tx_mm, ty_mm) that agree best with the pair measurements."""
    n = len(valid)
    idx = {s: i for i, s in enumerate(np.flatnonzero(valid))}
    m = len(idx)
    S = np.array([idx[s] for s, u, *_ in meas]); U = np.array([idx[u] for s, u, *_ in meas])
    d = np.array([u - s for s, u, *_ in meas], float)
    phi = np.array([p for _, _, p, _, _ in meas]); dxy = np.array([[dx, dy] for _, _, _, dx, dy in meas])
    w = 1.0 / d
    rows = np.arange(len(meas))
    A = np.zeros((len(meas) + 1, m))
    A[rows, S] = 1.0; A[rows, U] = -1.0
    A[-1, :] = 1.0                                   # gauge: mean rotation 0
    th, r_th, wr_th = _huber_lsq(A, np.r_[phi, 0.0], np.r_[w, 10.0])
    # translations with the rotations fixed
    vs = np.flatnonzero(valid)
    c = centroids_mm[vs]
    rhs = np.zeros((len(meas), 2))
    for k in range(len(meas)):
        a = np.radians(th[U[k]])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = c[S[k]] + dxy[k]
        rhs[k] = R @ (q - c[U[k]]) + c[U[k]] - c[S[k]]
    t = np.zeros((m, 2))
    for ax in range(2):
        t[:, ax], _, _ = _huber_lsq(A, np.r_[rhs[:, ax], 0.0], np.r_[w, 10.0])
    poses = np.zeros((n, 3))
    poses[vs, 0] = th
    poses[vs, 1:] = t
    return poses, dict(n_pairs=len(meas), rotation_residual_deg_median=float(np.median(np.abs(r_th[:-1]))),
                       downweighted_pairs=int((wr_th[:-1] < 1).sum()))


def posegraph_poses(stack, res_mm, thr, sigma=15.0, iters=2, distances=DEFAULT_DISTANCES, exclude=(), max_workers=1,
                    **_):
    """Poses (n x 3) and centroids (n x 2, voxels) of the pose-graph correction for a stack (x, y, slice);
    same convention as slice_jitter.jitter_poses (rotate about the centroid, then translate)."""
    n = stack.shape[2]
    valid = np.ones(n, bool)
    valid[[s for s in exclude if 0 <= s < n]] = False
    tissue = [ndi.binary_fill_holes(stack[..., s] > thr) for s in range(n)]
    valid &= np.array([t.sum() > 20 for t in tissue])
    centroids = np.array([ndi.center_of_mass(t) if t.sum() > 20 else (np.array(stack.shape[:2]) - 1) / 2
                          for t in tissue])
    total = np.zeros((n, 3))
    cur = stack
    logging.warning(f'Pose-graph correction: adjacent NCC median {np.median(adjacent_ncc(cur, thr)):.3f} before')
    for it in range(iters):
        c_mm = (centroids + total[:, 1:] / res_mm) * res_mm      # centroids of the current stack
        meas = pair_measurements(cur, valid, distances, thr, res_mm, max_workers=max_workers)
        p, info = solve_poses(meas, c_mm, valid)
        for k in range(3):
            p[:, k] -= _low_pass(np.where(valid, p[:, k], np.nan), sigma)[0]   # keep the slow path
        p[~valid] = 0.0
        total += p
        for k in range(3):
            total[:, k] -= _low_pass(np.where(valid, total[:, k], np.nan), sigma)[0]
        total[~valid] = 0.0
        cur = apply_poses(stack, total, centroids, res_mm)
        logging.warning(f'Pose-graph iteration {it + 1}/{iters}: {info}; adjacent NCC median '
                        f'{np.median(adjacent_ncc(cur, thr)):.3f}, rotation SD {total[:, 0].std():.2f} deg, '
                        f'shift SD {np.linalg.norm(total[:, 1:], axis=1).std():.3f} mm')
    return total, centroids
