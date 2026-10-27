# Slice registration: local setup, output contract, experiments

Branch `enh-local_uv_mri_anchor`. Started 2026-10-27. The original script
`run_slice_registration_optimized_newReg_sdf_v2.py` and the function library are **unchanged**;
all changes are in new files.

## 1. Environment (uv)

```bash
cd /home/chris/Documents/code/microscopy_scripts
uv sync                     # Python 3.12, antspyx 0.6.3 (see pyproject.toml / uv.lock)
```

- Python is pinned to 3.12: the script has no `if __name__ == '__main__'` guard and relies on the
  `fork` start method of `ProcessPoolExecutor` (3.14 changes the Linux default to `forkserver`).
- `nighres` (>=1.5) is not on PyPI and needs a Java/JCC build. It is **not** installed.
  `env_compat.py` provides the nighres functions that `embedded_antspy_smthMod.py` uses
  (`_output_dir_4saving`, `_fname_4saving`, `load_volume`, `save_volume`) and maps the
  ANTsPy <0.5 names `ants.utils._int_antsProcessArguments` / `ants.utils.get_lib_fn` to `ants.internal`.
- `env_compat.py` also sets 1 thread per worker and `oom_score_adj=1000` (our jobs are killed first
  on this shared machine).

## 2. Running the local copy

```bash
cd <scratch dir>   # ANTs can leave files in the cwd
SLICEREG_RESCALE=40 SLICEREG_NEIGHBOUR_TARGETS=coreg0nl SLICEREG_MAX_WORKERS=6 nice -n 19 \
  uv run --project /home/chris/Documents/code/microscopy_scripts \
  python /home/chris/Documents/code/microscopy_scripts/macaque_CB/run_slice_registration_optimized_newReg_sdf_v2_local.py
```

| Variable | Default | Meaning |
|---|---|---|
| `SLICEREG_RESCALE` | 5 | in-plane downsampling of the 10 µm images (5 → 50 µm, 40 → 400 µm) |
| `SLICEREG_MAX_WORKERS` | 8 | parallel registrations |
| `SLICEREG_NEIGHBOUR_TARGETS` | `coreg0nl` | STAGE1 neighbour targets, see §4 |
| `SLICEREG_SMOKE` | 0 | 1 = first 7 slices, slice 3 as missing, rescale 40 |
| `SLICEREG_THREADS_PER_WORKER` | 1 | ITK/OpenMP threads per worker |
| `SLICEREG_MRI_ANCHOR` | 0 | 1 = MRI planes replace the stack-median target in STAGE1 (§4.2) |
| `SLICEREG_MRI_WEIGHT` | 1.0 | weight factor of the MRI target |
| `SLICEREG_MRI_REFIT` | 0 | 1 = refit the MRI affine at every STAGE1 level (drifts, see 4.2); 0 = one fit on `cascade_0` |
| `SLICEREG_MRI_DIRECTION` | 1 | D99 hemisphere the stack is fitted to (+1 right, -1 left); fixed so outputs are comparable |
| `SLICEREG_JITTER` | 0 | 1 = rigid jitter correction folded into coreg0nl (4.7); `SLICEREG_JITTER_SIGMA` (15), `SLICEREG_JITTER_ITERS` (6) |
| `SLICEREG_STAGE1_SYN` | 1 | 0 = STAGE1 rigid only |
| `SLICEREG_STAGE1_SYN_LAST` | 1 | 0 = SyN only in the template-building STAGE1 levels, the last level rigid |
| `SLICEREG_STAGE1_REG` | Medium | STAGE1 SyN regularization preset |
| `SLICEREG_EXPORT_ZARR` | 1 | export the final densities as OME-Zarr at the end of the run (§5) |

Data: the lookup csv lists `/data/data_drive/...`; on this workstation that is `/mnt/ssd/...`
(the script rewrites the prefix). Output:
`/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/zefir_sliceReg_optimized_v2_rescale_{rescale}_local{variant}/`
(variant: `''` = original behaviour, `_nbrprev` = previous-level neighbours).

## 3. Output contract (used to push other cell data into the final space)

Nothing in this section has changed. Any future change must be listed here.

- Native slices: `{subject}_{idx:04d}_{img}.nii.gz`, made with `downsample_image(img, rescale, prop_pad=0.2)`
  and `create_affine(shape, voxel_res=[10*rescale, 10*rescale, 50])` (µm, centred).
- Deformed slices: `{subject}_{idx:04d}_{img}_{level}_ants-def0.nii.gz`;
  maps: `..._{level}_ants-map.nii.gz` (forward), `..._{level}_ants-invmap.nii.gz` (inverse),
  coordinate-mapping images as written by `embedded_antspy_2d_multi`.
- `{subject}_forward_maps.csv` / `{subject}_inverse_maps.csv`: one row per slice, one column per level.
- Transform chain as built by `get_transform_chain_for_level` (this is what the code does):
  - `coreg0nl` → `rigsyn_iter4_win12` (STAGE1 always registers **from** `coreg0nl`, so each
    `rigsyn_*` map is a single step from `coreg0nl`; `cascade_0` and earlier `rigsyn` maps are **not** chained)
  - groupwise: `coreg0nl` → `rigsyn_iter4_win12` → `groupwise_iter3` → … → `groupwise_iterN`
  - Note: the README table lists `cascade_0` and all `rigsyn` iterations as chain steps; the code above is correct.
- Missing slices (`missing_idxs_to_fill = [32,59,120,160,189,228]`): the `_ants-def0` files at each level are
  overwritten with interpolated slices; originals are copied to `missing_slices/`.

## 4. Changes and experiments

### 4.1 STAGE1 neighbour targets (opt-in, `SLICEREG_NEIGHBOUR_TARGETS=previous`)

Finding: in the original code the neighbour targets in every STAGE1 iteration are the translation-only
`coreg0nl` slices (confirmed in the run log). Only the per-slice template and the stack median improve
between iterations. The neighbours carry 54% (pass 1) and 77% (pass 2) of the metric weight.

Change: `slice_registration_extensions.run_parallel_coregistrations_v3(neighbour_target_tag=...)`.
With `previous`, the neighbours come from the level that also gives the template (`cascade_0` for
iteration 0, then the previous STAGE1 output; pass 2 uses the pass 1 output). The **source** is still
`coreg0nl`, so outputs, maps and the transform chain are unchanged (§3).

Expected: better local continuity; risk of more low-frequency drift (the translation-only neighbours
acted as a weak anchor to the raw positions).

| Run | Rescale | Neighbours | Status / result |
|---|---|---|---|
| A | 40 | `coreg0nl` (original) | done (1 h 47 min, 6 workers) |
| B | 40 | `previous` | done (1 h 47 min, 6 workers) |
| C | 40 | `previous` + MRI anchor, refit every level | done; MRI fit drifted, see 4.2 and below |
| C2 | 40 | `coreg0nl` + MRI anchor, affine fixed at `cascade_0` | running |

Result A vs B (`evaluate_registration.py`, MRI fit with direction +1):

| measure | A | B |
|---|---|---|
| adjacent NCC median, stage1 / groupwise | 0.855 / 0.944 | 0.853 / 0.941 |
| adjacent NCC 5th pct, stage1 / groupwise | **0.667 / 0.841** | 0.599 / 0.804 |
| MRI per-slice Dice median, stage1 / groupwise | **0.861 / 0.865** | 0.825 / 0.772 |
| centroid offset to MRI plane, median mm, stage1 / groupwise | **1.46 / 1.39** | 2.48 / 2.96 |
| STAGE1 warp RMS total / non-affine (mm) | **0.59 / 0.27** | 1.43 / 0.31 |

**Correction (same day):** the MRI-based numbers above used a tissue threshold of 2 counts per voxel, which at
rescale 40 (64x the pixel area of rescale 5) made tissue masks 25-45% too large. With the density-based
threshold (`mri_anchor.tissue_threshold_for`) and the more robust shear-free affine:

| run | NCC median s1 / gw | NCC 5th pct s1 / gw | MRI Dice slice median s1 / gw | offset median mm s1 / gw | offset p95 mm s1 / gw | warp total / non-affine mm |
|---|---|---|---|---|---|---|
| A | 0.796 / 0.914 | **0.485 / 0.724** | 0.846 / 0.857 | 1.09 / 1.11 | 3.28 / 3.70 | **0.59 / 0.25** |
| B | 0.797 / 0.907 | 0.402 / 0.675 | 0.835 / 0.849 | 1.16 / 0.93 | 1.65 / 1.73 | 1.43 / 0.29 |
| C | 0.778 / 0.897 | 0.312 / 0.636 | 0.829 / 0.838 | 1.18 / 1.22 | 3.48 / 3.18 | 1.77 / 0.28 |

"B drifts more" is **not** supported any more: the MRI-based measures carry fit noise of the same size as the
differences (the in-plane scale of repeated fits ranged 0.64-0.88). The decision below rests on the measures that
need no MRI: the continuity tail and the warp size.

Decision: keep the original neighbour targets (`coreg0nl`). The translation-only neighbours act as an anchor to
the raw section positions; without them the stack drifts more and warps more, with no gain in continuity.
Run C (B + MRI, refit at every level): per-slice MRI Dice 0.873 / 0.867, centroid offset 1.47 / 1.51 mm (back to
A's level, so the anchor removed B's drift), but adjacent NCC 5th pct 0.516 / 0.789 and STAGE1 warp RMS 1.79 mm
(worst of the three). Slice 307 failed in the last STAGE1 iteration (MI with its template 0.74 -> 0.23 / 0.13,
the slice slid out of the field of view): its MRI plane came from the drifted fit (section scale 1.2) and was far
more medial than the slice. QC idea: flag slices whose MI in `{subject}_MI_values.csv` drops sharply.

A still shows low-frequency drift (offset rises to ~2.5 mm near slice 180 and falls to ~0.1 mm near slice 235),
which is what the MRI anchor is for. (The offset includes a bias from the partial brainstem; compare runs, not
absolute values.)

Compare runs with `evaluate_registration.py` (adjacent-slice NCC, independent MRI fit per final stack with
per-slice Dice and centroid offset, non-affine warp size of the final STAGE1 maps).

### 4.2 MRI anchor (D99 cerebellum + brainstem, 0.25 mm) — prototype

Target files: `local_extras/cerebellum_target/` (git-ignored). Prototype on the existing rescale-5
`groupwise_iter9` stack:

1. Downsample the stack to 0.25 mm.
2. 2D orientation search on one section vs. the D99 sagittal plane (rotation × flip, MI):
   in-plane flip + ~74° rotation, in-plane scale ~0.85. The two mirror sides score nearly the same
   (symmetric anatomy), so the hemisphere label does not matter.
3. 3D registration to D99 (masked to our side + 1 mm): affine, then coarse SyN.

| Stage | 3D Dice (our side) | Median per-plane Dice |
|---|---|---|
| orientation only | 0.78 | 0.84 |
| + affine | 0.84 | 0.85 |
| + coarse SyN | 0.88 | 0.88 |

Facts from the fits: sections are sagittal, one hemisphere + vermis + ~3 mm past the midline;
histology is ~15% smaller in-plane than D99 but close to D99 scale through-plane (consistent with
in-plane shrinkage after mounting), so the 3D step must be affine, not similarity.

Pipeline integration (`SLICEREG_MRI_ANCHOR=1`, `SLICEREG_MRI_WEIGHT`, module `mri_anchor.py`):

- at every STAGE1 template level (`cascade_0`, `coreg12nl_rigsyn_k`, `coreg12nl_win12_rigsyn_k`) the stack is
  fitted to the MRI (first level: area fit + orientation search for both slice directions; later levels:
  warm start from the previous fit) and D99 is resampled into the stack grid
- the 3D affine is projected to rotation x per-axis scale (no shear): a shear between the slice axis and the
  in-plane axes is what a linear drift looks like, so it must not be copied into the targets. Cost: Dice 0.807 -> 0.805
- fitted scales (MRI -> stack, rescale-5 stack): in-plane 0.86 / 0.87, slice axis 0.98 (~51 um of D99 per section)
- each slice's MRI plane replaces the whole-stack median target (same weight x `SLICEREG_MRI_WEIGHT`)
- new files only: `*_{level}_mri_target.nii.gz`, `mri_anchor/{level}/` (transforms, `qc.json`).
  Names in §3 are unchanged.
- Run C (refit at every level, warm start) drifted: Dice of the fit 0.86 (cascade_0) -> 0.76, section-axis scale
  1.01 -> 1.20 by iteration 1, then stable. The reference followed the stack. Probable reasons: the refit itself,
  and the MRI target also drives the SyN part (embedded_antspy_2d_multi uses one target set for rigid and SyN),
  so at rescale 40 slices are warped toward the D99 anatomy (another animal). Default is now one fit on
  `cascade_0`, kept for all levels (`SLICEREG_MRI_REFIT=0`). If C2 still shows the SyN effect, the next step is
  to use the MRI in a rigid-only per-slice step (with the poses smoothed along the stack) instead of as a SyN target.

### 4.3 Missing-slice interpolation check (rescale-5 `groupwise_iter9` stack)

| slice | NCC prev-next | NCC synthetic-neighbours | note |
|---|---|---|---|
| 59, 189 | 0.91-0.94 | ~0.97 | good |
| 120, 228 | 0.69-0.82 | 0.87-0.94 | acceptable |
| 160 | 0.61 | ~0.85 | neighbours disagree |
| 32 | 0.41 | ~0.76 | neighbours disagree strongly (pair 33-34 is also among the worst in the stack) |

Synthetic slices keep 80-93% of the edge sharpness of their neighbours. Where they are weak, the cause is
poor alignment of slices i-1 and i+1 across the 100 um gap, not the interpolation itself.

### 4.4 Is the MRI-derived low-frequency correction real? (offline test, `mri_pose_consistency.py`)

Two equally good global fits of the same stack to D99 (reference slices 140 and 260), per-slice rigid fit
(mask-based) of every slice to its D99 plane, poses smoothed along the stack (sigma 10 / 20 slices).

| stack | global fits (Dice; in-plane scales) | cross-fit r: rotation / dx / dy | after removing a linear trend |
|---|---|---|---|
| A, rescale 40 | 0.845, 0.860; 0.76/0.71, 0.77/0.66 | 0.42-0.53 / 0.58-0.61 / **-0.75 to -0.78** | 0.16-0.23 / 0.68-0.73 / 0.17 |
| rescale-5 groupwise, fitted at 0.1 mm | 0.838, 0.838; 0.75/0.69, 0.77/0.67 | ~0 / 0.71-0.78 / **-0.76 to -0.79** | 0.08-0.16 / 0.71-0.78 / 0.37-0.44 |

Even/odd slices of one fit agree (r > 0.98), so the per-slice fitting is not noisy; but each global fit puts
its own slow trend into the corrections (dy trends of opposite sign, rotation ramps of 20-40 deg in one fit only).
With a different-animal MRI, a linear drift and a small difference in the global tilt are not separable.
Only the in-plane component along stack axis i (mostly superior-inferior) is partly consistent (r ~0.75; both fits
show a ~1 mm dip near slice 220, where A's offset curve has its bump).

**Conclusion:** do not build per-slice MRI anchoring with D99. Use D99 for global placement (OME-Zarr `mri`
placement) and QC only. The resolution is not the cause (same result at rescale 40 and at 0.1 mm).

### 4.5 MRI-free drift control (offline test, `drift_control_test.py`)

Raw slices (`coreg0nl`) are the separate scans placed by their image centres (the initial step has no
registration). Rigid part of each slice's STAGE1 map (raw -> final), fitted over tissue, smoothed along the stack:

| run | rotation slow SD, sigma 15 (expected if random) | slow translation SD dx / dy |
|---|---|---|
| A | 0.53 deg (0.39) | **0.05 / 0.03 mm** |
| B | 0.91 deg (0.42) | 0.76 / 0.80 mm |
| C | 0.65 deg (0.42) | 1.00 / 0.78 mm |

- Rotation: A's slow rotation is close to what independent mounting angles leave (little rotational drift);
  B has twice that (drift). Removing A's slow rotation (max 3.5 deg) changes adjacent NCC slightly for the better
  (median 0.796 -> 0.810, 5th pct 0.485 -> 0.506). Small effect; not added to the pipeline.
- Translation cannot be controlled this way: the raw scans are framed on the tissue (bounding-box centre at the
  image centre, SD 0.3 mm, slow part 0.07 mm). A keeps this framing, so A's stack has every section's bounding-box
  centre on one line. In D99 the bounding-box centre of the sagittal planes across our hemisphere moves 2.5-6.6 mm
  (1.6-3.4 mm after removing a linear trend), so A can straighten the true shape by a few mm ("banana effect" from
  the framing). B and C drift away from A by 2-4 mm, mostly in TP3, growing along the stack (chained drift); the
  MRI fits do not confirm their shifts (r -0.06 to 0.41 for dx; dy follows whichever fit's linear trend).
- So the remaining low-frequency uncertainty is the in-plane translation path, and the data hold no reference for
  it (no same-animal MRI). Options: a physical reference if one exists (blockface photos, fiducials), or accept A
  and document the limitation.

### 4.6 Decision (2026-10-27)

Keep run A's setup: original neighbour targets (`coreg0nl`), no MRI term in STAGE1 (`SLICEREG_MRI_ANCHOR=0`),
D99 for global placement (OME-Zarr `mri` placement) and QC only. Known limit: the slow in-plane translation
path follows the scan framing (bounding-box centred sections), possibly a few mm off the true shape across the
hemisphere (4.5). These are the defaults of the local script.

### 4.7 Improving the alignment without MRI (started 2026-10-27)

Diagnosis of A (rescale 40): A corrects slice-to-slice jitter (fast part of its STAGE1 corrections: rotation SD
2.8 deg, shift SD ~0.3 mm) but keeps the slow path of the scan framing. Weak adjacent pairs have larger jumps in
tissue area and centroid (Spearman r ~ -0.25): partly real tissue change or damage. A's final stack shows
intensity bands (staining; left as they are in the outputs) and jagged edges mostly in TP3.

**Measurement caveat:** adjacent NCC rises with interpolation blur. A half-pixel resampling alone took A's STAGE1
median from 0.80 to 0.88. Compare runs with the same number of resampling steps, or use the blur-equalized NCC
in `evaluate_registration.py` (same in-plane smoothing for every stack), plus adjacent tissue-mask Dice.

Rigid jitter correction (`slice_jitter.py`, offline test `jitter_correction_test.py`): each slice fitted rigidly
(GC) to the mean of its neighbours, the slow part of the poses removed (sigma 15 slices), half steps, iterated.
- on A's STAGE1 / final stacks (offline, resampled twice): NCC gain over a sub-pixel-shift control of about
  +0.04 / +0.02 (median) and +0.10 / +0.04 (5th pct); mask Dice unchanged.
- folded into coreg0nl (one interpolation of the native slice, maps/inverse maps rewritten in place, names and
  chain unchanged): raw coreg0nl adjacent NCC 0.41 -> 0.74; rotation SD 5 deg, shift SD 0.2 mm.
- on the rescale-5 final stack (fitted at 0.1 mm), against a sub-pixel-shift control: NCC median 0.820 -> 0.836,
  5th pct 0.414 -> 0.476; blur-equalized 5th pct 0.583 -> 0.627; mask Dice unchanged; rotation SD 0.5 deg,
  shift SD 0.05 mm. Small but real gain in the weakest pairs, even after the groupwise step.
- pipeline switch `SLICEREG_JITTER=1` (output suffix `_jit`); poses in `{subject}_coreg0nl_jitter_poses.csv`.

| Run | Rescale | Change | Status |
|---|---|---|---|
| J | 40 | A + jitter correction in coreg0nl | done: better than A (below) |
| JR | 40 | J + STAGE1 rigid only (`SLICEREG_STAGE1_SYN=0`; all warping in the groupwise step) | done: mixed, worse than J |
| JT | 40 | J + SyN only in the template-building STAGE1 levels 0-3, last level (in the chain) rigid (`SLICEREG_STAGE1_SYN_LAST=0`) | running |

Results at rescale 40 (`evaluate_registration.py --no-mri --reference A`; median / 5th pct of adjacent pairs):

| Measure | A STAGE1 | J STAGE1 | JR STAGE1 | A final | J final | JR final |
|---|---|---|---|---|---|---|
| adjacent NCC | 0.796 / 0.485 | **0.871 / 0.639** | 0.816 / 0.553 | 0.914 / 0.724 | **0.936 / 0.796** | 0.922 / 0.752 |
| blur-equalized NCC | 0.936 / 0.735 | **0.961 / 0.801** | 0.911 / 0.723 | 0.968 / 0.839 | **0.977 / 0.874** | 0.959 / 0.819 |
| tissue-mask Dice | 0.964 / 0.914 | **0.973 / 0.922** | 0.957 / 0.884 | 0.976 / **0.939** | **0.982** / 0.938 | 0.976 / 0.921 |
| STAGE1 warp RMS, total / non-affine (mm) | 0.59 / 0.25 | 0.32 / 0.20 | 0.15 / 0.00 | | | |
| slow path vs A, centroid max (mm) | | 0.17 | 0.48 | | 0.08 | 0.35 |

- J beats A on every measure except final mask-Dice 5th pct (equal, 0.938 vs 0.939), with about half the STAGE1
  warp and the same slow path (centroid within 0.1 mm of A in the final stack). Through-plane views: folia layers
  more continuous, no edge artifacts. Per-slice total counts at coreg0nl are unchanged by the correction.
- JR (no STAGE1 SyN) has the smallest warp, but lower blur-equalized NCC and mask Dice than A: the STAGE1 SyN
  corrects real shape differences between sections, not only noise. Single misfit slices show as dark lines.

Note on what STAGE1 warps do: every STAGE1 level registers from coreg0nl again, so the warps of levels 0-3 (and of
the cascade) only shape the templates. Only the last level (`rigsyn_4`) feeds the groupwise step and is in the chain.

Bug found in the first JR run (fixed in `slice_jitter.py`, commit 434b3c8): at the border of the valid map region the
composed map was interpolated with the zeros outside it, which copied tissue to the image edges (dotted streaks).
That first JR result (worse mask Dice, a false 50 deg slow-angle difference at the lateral end) is void; the first J
run was stopped. After the fix, raw coreg0nl adjacent NCC with the correction is 0.78 (0.74 with the bug).
The slow-path angle in `evaluate_registration.py` now skips slices with a near-round outline (ill-defined axis).

## 5. OME-Zarr export of the final densities

`export_ome_zarr.py` (our env) writes the final stack as OME-Zarr **NGFF 0.6 on Zarr v3**, through
`neuralarc.core.ome_zarr.write_ome_zarr` from tck_view, run in tck_view's own uv environment
(`_tckview_write_ome_zarr.py`; tck_view needs Python >= 3.13 and vtk/trame, so it is not a dependency here).
The pipeline calls it at the end of a run (`SLICEREG_EXPORT_ZARR=1`):
`{output_dir}{subject}_{final_level}_density_{stack|mri}.ome.zarr`.

- Values: **cells/mm^3** = pipeline counts per voxel / voxel volume (the stacks hold counts: `downsample_image`
  sums blocks). The factor and the units are in the attributes under `microscopy_scripts`.
- Pyramid: tck_view's default (mean; the finest axes halve first, then all, until the largest side <= 256;
  64^3 chunks, Blosc zstd). Rescale 5: (398, 873, 623) -> (199, 437, 312) -> (100, 219, 156) (z, y, x).
- Labels: `labels/synthetic_slices` = 1 on the interpolated slices (not measured).
- Our attributes are under `neuralarc.microscopy_scripts` in the root `zarr.json` (tck_view merges them there).
- Placement `stack`: the registered grid in mm, no anatomical claim.
  Placement `mri`: the same voxels (not resampled) with an NGFF 0.6 affine to D99 world (RAS, mm), from a fresh
  shear-free fit of the stack to D99 (`--direction`, default +1 = right hemisphere).
  The array is permuted and flipped (not resampled) to the closest RAS+ order before writing: tck_view's slice
  panes take array axes 0, 1, 2 as world x, y, z, and with the stack order (in-plane, in-plane, section) the axial
  pane showed the wrong plane (reported 2026-10-27). The section axis is then array x; the mapping is stored as
  `array_axes_from_stack` (row k: stack axis k (i, j, section) is array axis [0], reversed if [1] == -1).
- Open in tck_view: `uv run --project ~/Documents/code/tck_view arc <file>.ome.zarr`.

Exports of the existing rescale-5 `groupwise_iter9` stack (made before this branch):
`/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/ome_zarr_exports/zefir_r5_gw9_{stack,mri}.ome.zarr`

## 6. Comparison: BigBrain reconstruction (Lepage, Mohlberg et al., BigBrain2 workshop abstracts)

- MRI: the post-mortem MRI of the **same** fixed brain (1.5T MPRAGE, 0.5 mm) - an undistorted reference of that
  brain, which also gives the shrinkage. We have D99, another animal.
- Order: (1) iterative alignment to the MRI, alternating 3D stack->MRI and 2D section->sliced MRI, rigid -> affine
  -> non-linear over 10 global iterations (low-frequency errors, "jaggies", shrinkage); (2) then, separately,
  section-to-section non-linear 2D alignment **without MRI** (high-frequency errors); intensity balancing to MRI.
- The first BigBrain (2013) did section-to-section first and registered the volume to MNI afterwards; low-frequency
  jaggies remained, which motivated the MRI-first scheme.
- Lessons for us: keep the MRI and the neighbour terms in separate steps (run C mixed them in one cost, with the MRI
  also driving SyN); with a different-animal MRI stop at rigid (+ a constrained scale), and smooth the poses along
  the stack, so that only low-frequency corrections come from the MRI.
