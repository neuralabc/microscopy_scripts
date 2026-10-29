# Slice registration: local setup, output contract, experiments

Branch `enh-local_uv_mri_anchor`. Started 2026-10-27. The original script
`run_slice_registration_optimized_newReg_sdf_v2.py` and the function library are **unchanged**;
all changes are in new files.

Short guide to all scripts: `SCRIPTS_local.md`. HPC (full resolution): `run_slice_registration_optimized_newReg_sdf_v2_hpc.py`.
Lobule labels (work in progress): `LOBULE_LABELS_HANDOFF.md`.

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
| `SLICEREG_JITTER` | 1 (since 2026-10-28; was 0) | 1 = rigid per-slice pose correction folded into coreg0nl (4.7, 4.8); 0 = the original pipeline (no suffix); `SLICEREG_JITTER_SIGMA` (15), `SLICEREG_JITTER_ITERS` (2 for posegraph, 6 for jitter) |
| `SLICEREG_JITTER_METHOD` | posegraph (since 2026-10-28; was jitter) | `posegraph` (§4.8, suffix `_pg`) or `jitter` (nearest neighbours, §4.7, suffix `_jit`) |
| `SLICEREG_GROUPWISE_ITERS` | 10 | number of groupwise iterations; the final level becomes `..._groupwise_iter{N-1}` (suffix `_gw{N}`; jitter iterations other than 6 add the number to `_jit`) |
| `SLICEREG_STAGE1_SYN` | 1 | 0 = STAGE1 rigid only |
| `SLICEREG_STAGE1_SYN_LAST` | 1 | 0 = SyN only in the template-building STAGE1 levels, the last level rigid |
| `SLICEREG_STAGE1_REG` | Medium | STAGE1 SyN regularization preset |
| `SLICEREG_EXPORT_ZARR` | 1 | export the final densities as OME-Zarr at the end of the run (§5; `stack` placement only since 2026-10-28) |
| `SLICEREG_QC` | 1 | write the QC report `<output_dir>/QC/index.html` at the end of the run (§7) |
| `SLICEREG_MRI_REGISTER` | 1 (since 2026-10-28) | run `mri_register.py` (reference affine + SyN + D99 labels) on the final level at the end of the run; skipped with a message if the D99 files are not found |
| `SLICEREG_COUNTS_STACK` | 1 (since 2026-10-29) | write the count-preserving stack of the final level (`count_preserving_stack.py`, §7.4) before the MRI step; `SLICEREG_COUNTS_WORKERS` sets its workers when it is run by hand (default 8; the run scripts pass `SLICEREG_MAX_WORKERS`) |
| `SLICEREG_D99_DIR` | `local_extras/cerebellum_target/` (HPC script: `/data/neuralabc/johmat/phase_ml/D99/D99_v2.0_dist/cerebellum_target`) | folder of the D99 target files, for all scripts |
| `SLICEREG_TCK_VIEW` | `~/Documents/code/tck_view` | tck_view checkout for the OME-Zarr export; no checkout = export skipped with a message |
| `SLICEREG_LOOKUP_CSV`, `SLICEREG_OUTPUT_ROOT` | HPC script only | input lookup csv (its image paths are used as they are), output root (`/tmp/`) (see `SCRIPTS_local.md`) |

Each run also writes `run_settings.json` (all `SLICEREG_*` variables, git commit, main parameters) in its output
folder; the QC report shows it.

Data: the lookup csv lists `/data/data_drive/...`; on this workstation that is `/mnt/ssd/...`
(the script rewrites the prefix). Output:
`/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/zefir_sliceReg_optimized_v2_rescale_{rescale}_local{variant}/`
(variant: `''` = original behaviour, `_nbrprev` = previous-level neighbours).

## 3. Output contract (used to push other cell data into the final space)

File names, formats and the chain below have not changed. Any future change must be listed here.

Changes in content (2026-10-28), with the per-slice pose correction now on by default (`SLICEREG_JITTER=1`,
method `posegraph` since the evening of 2026-10-28; first `jitter`):
- The output folder has the suffix `_pg` (e.g. `..._rescale_5_local_pg/`, `..._rescale_5_hpc_pg/`); `_jit` for the
  jitter method. Both methods write the same files in the same format.
- The `coreg0nl` maps, inverse maps and deformed slices include a rigid per-slice correction (rotation about the
  tissue centroid + shift). The chain still starts at `coreg0nl`, so pushing data through the chain needs no change.
  A tool that builds its own `coreg0nl` step (image-centre placement) instead of reading the `coreg0nl` maps must
  apply the poses in `{subject}_coreg0nl_jitter_poses.csv` (theta_deg, tx_mm, ty_mm, centroid_i_vox,
  centroid_j_vox per slice; `slice_jitter.pose_to_voxel` gives the voxel form).

New files (2026-10-28), nothing existing changes:
- `{subject}_{final_level}_tissue_mask.nii.gz` beside the final stack: uint8, 1 = tissue, same grid, affine and
  header as the stack. Tissue = counts > 2 per 50 um voxel (about 16,000 cells/mm^3; the count threshold scales
  with the voxel area, so it means the same at every rescale), holes filled in each slice. This is the definition
  that the QC, the evaluation and the MRI fit use. For another level:
  `uv run python macaque_CB/tissue_mask.py --run-dir <dir> --rescale <r> --level <level>`.
- `run_settings.json`, `QC/`, and (from `mri_register.py`) `mri_registration/<level>/`.
- Cleanup (`cleanup_run.py`, optional, run by hand): keeps the native sections, the forward and inverse maps of
  the chain below (from `{subject}_forward_maps.csv` via `get_transform_chain_for_level`), the final-level `def0`
  sections and all run-level files; removes the maps of other levels, other deformed sections, per-section
  templates and ANTs intermediate files, and lists them in `cleanup_manifest.txt`. The maps csv files are not
  changed, so their columns for removed levels point to absent files. Tested 2026-10-28: P20 4.0 of 6.3 GB
  removable; after a cleanup (smoke run) the kept chain reproduces the final `def0` sections exactly (NCC 1.000).
  Procedure, the files of a push, and what a cleanup changes: `SCRIPTS_local.md` §7.
- From `mri_register.py` (via `atlas_labels.py`, §7.3), beside the stack of the registered level:
  `{subject}_{level}_D99_labels.nii.gz` (uint8: D99 v2.0 label value in tissue, 0 outside tissue, 252 / 253 =
  D99 cerebellum / brainstem without a label, 254 = in the D99 mask but in neither region, 255 = tissue outside
  the D99 mask), `{subject}_{level}_D99_regions.nii.gz` (uint8: 1 vermis, 2 hemisphere cortex, 3 deep cerebellar
  nuclei, 4 other D99 label, 252-255 as above),
  `{subject}_{level}_D99_x_mm.nii.gz` (float32, D99 world x in mm) and `{subject}_{level}_D99_labels.tsv` (cells
  per label). Same grid, affine and header as the stack. Label names:
  `mri_reference/D99_v2.0_labels_cerebellum_brainstem.tsv`.

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

**Correction (2026-10-28): this test is void.** In both stacks, fit 1 was placed turned by about 165 deg (about stack
axis i) against the checked reference (§7.2); fit 2 agrees with the reference (1 deg / 2.5 deg). MI shows it
(on A's STAGE1 stack: fit 1 -0.063, fit 2 -0.101, reference -0.096). A turn about axis i keeps the sign of dx and
reverses dy, which is the pattern above (dx r +0.6..0.8, dy r -0.76). So the test compared a correct and a wrong
placement; it does not show that D99 cannot constrain the slow path. Run C (§4.2) is still a valid negative
result for MRI targets in SyN and for a refit at every level.

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
| JT | 40 | J + SyN only in the template-building STAGE1 levels 0-3, last level (in the chain) rigid (`SLICEREG_STAGE1_SYN_LAST=0`) | done: same as JR, worse than J |
| A20, J20 | 20 | A and J at rescale 20: does the gain hold at a finer grid? | done: yes, larger gain (below) |
| JG | 40 | J + 10 jitter iterations + 15 groupwise iterations (`_jit10_gw15`): convergence and warp per iteration | done (below) |

Results at rescale 40 (`evaluate_registration.py --no-mri --reference A`; median / 5th pct of adjacent pairs):

| Measure | A STAGE1 | J STAGE1 | JR STAGE1 | A final | J final | JR final |
|---|---|---|---|---|---|---|
| adjacent NCC | 0.796 / 0.485 | **0.871 / 0.639** | 0.816 / 0.553 | 0.914 / 0.724 | **0.936 / 0.796** | 0.922 / 0.752 |
| (JT: NCC STAGE1 0.812 / 0.532, final 0.921 / 0.739; blur-eq final 0.959 / 0.817; Dice final 0.975 / 0.921; warp 0.10 / 0.00; slow path vs A 0.28 mm) | | | | | | |
| blur-equalized NCC | 0.936 / 0.735 | **0.961 / 0.801** | 0.911 / 0.723 | 0.968 / 0.839 | **0.977 / 0.874** | 0.959 / 0.819 |
| tissue-mask Dice | 0.964 / 0.914 | **0.973 / 0.922** | 0.957 / 0.884 | 0.976 / **0.939** | **0.982** / 0.938 | 0.976 / 0.921 |
| STAGE1 warp RMS, total / non-affine (mm) | 0.59 / 0.25 | 0.32 / 0.20 | 0.15 / 0.00 | | | |
| slow path vs A, centroid max (mm) | | 0.17 | 0.48 | | 0.08 | 0.35 |

- J beats A on every measure except final mask-Dice 5th pct (equal, 0.938 vs 0.939), with about half the STAGE1
  warp and the same slow path (centroid within 0.1 mm of A in the final stack). Through-plane views: folia layers
  more continuous, no edge artifacts. Per-slice total counts at coreg0nl are unchanged by the correction.
- Known data feature (not a registration error): slices 277-285 (TP3 Image_13_02 - Image_17_02) have extra tissue
  at the upper edge (spike in through-plane views near y=53), already in the raw coreg0nl stack; the rest of each
  outline agrees with the neighbours. Folium crest or tissue flap, not decided. Slices 272-276 are torn at the
  upper right (area and centroid step at 277). Within-slide section order (_01/_02) checked: correct here.
- JR (no STAGE1 SyN) has the smallest warp, but lower blur-equalized NCC and mask Dice than A: the STAGE1 SyN
  corrects real shape differences between sections, not only noise. Single misfit slices show as dark lines.
- JT (SyN only for the templates, rigid last level) gives the same result as JR: better templates do not help
  when the slice that enters the chain is only rigidly aligned. The STAGE1 SyN in the chain is needed.
- Decision candidate: J (`SLICEREG_JITTER=1`, STAGE1 unchanged). Confirm at a finer rescale before the full run.

Convergence (rescale 40):
- Jitter iterations: raw coreg0nl NCC 0.807 at 6, 0.813 at 9-10 (flat). 6 is nearly enough; more than 8-10 adds nothing.
- STAGE1: plateau at `rigsyn_1`-`rigsyn_2` in A and J; levels 3-4 add nothing (but they are cheap at this point).
- Groupwise iterations 0-2 do not change the stack (not chained). From iteration 3 the weakest pairs improve at
  every iteration, with no plateau at 15 (JG, blur-equalized NCC 5th pct): iter9 0.883, iter14 0.904; Dice 5th pct
  0.935 -> 0.946. The warp per iteration is small and almost constant (total RMS 0.020-0.023 mm, non-affine 0.013 mm,
  voxel 0.4 mm).
- Cost of more iterations (independent check, per-slice total counts of the final stack / native slice; a change
  means the warp stretched or compressed the slice): JG iter9 -> iter14, median ratio 0.992 -> 0.986, median |dev|
  1.7% -> 2.0%, 95th pct 7.7% -> 7.9%. The groupwise step slowly shrinks the slices (about 0.1% per iteration).
  For comparison, A iter9: median ratio 0.982, |dev| 2.5% / 7.5%; J iter9: 0.989, 1.8% / 7.7%.
- Through-plane views of JG iter14 and J iter9: very similar; slightly smoother, no visible loss of structure.
- Bug in the original groupwise function from iteration 11 on (file names); fixed in the v2 copy (commit fb06be9).

Rescale 20 (0.2 mm, 2026-10-28; A20 vs J20, 10 groupwise iterations):

| Measure | A STAGE1 | J STAGE1 | A final | J final |
|---|---|---|---|---|
| adjacent NCC | 0.729 / 0.360 | **0.855 / 0.487** | 0.889 / 0.631 | **0.932 / 0.708** |
| blur-equalized NCC | 0.856 / 0.511 | **0.926 / 0.622** | 0.932 / 0.702 | **0.959 / 0.776** |
| tissue-mask Dice | 0.957 / 0.901 | **0.970 / 0.910** | 0.974 / 0.925 | **0.979 / 0.934** |
| STAGE1 warp RMS, total / non-affine (mm) | 0.58 / 0.25 | 0.29 / 0.21 | | |
| per-slice count ratio, median / median abs dev / 95th pct | | | 0.978 / 2.6% / 6.6% | 0.985 / 2.0% / 7.4% |
| slow path vs A, centroid max (mm) | | 0.36 | | 0.20 |

- The gain is larger than at rescale 40: final blur-equalized residual (1 - NCC) -40% at the median, -25% at the
  5th pct (rescale 40: -28% / -22%). J wins every measure except the 95th pct of the count deviation (7.4% vs 6.6%).
- Through-plane views: folia clearly more continuous in J, also in the lateral part (slices 250-399).
- Decision (2026-10-28): J is the default (`SLICEREG_JITTER=1`; set 0 for the original pipeline).


Note on what STAGE1 warps do: every STAGE1 level registers from coreg0nl again, so the warps of levels 0-3 (and of
the cascade) only shape the templates. Only the last level (`rigsyn_4`) feeds the groupwise step and is in the chain.

Bug found in the first JR run (fixed in `slice_jitter.py`, commit 434b3c8): at the border of the valid map region the
composed map was interpolated with the zeros outside it, which copied tissue to the image edges (dotted streaks).
That first JR result (worse mask Dice, a false 50 deg slow-angle difference at the lateral end) is void; the first J
run was stopped. After the fix, raw coreg0nl adjacent NCC with the correction is 0.78 (0.74 with the bug).
The slow-path angle in `evaluate_registration.py` now skips slices with a near-round outline (ill-defined axis).

### 4.8 Blocks of sections at a common angle; pose-graph correction (2026-10-28)

Reported by eye in J20 (QC through-plane cut and a 3D rendering): steps in the surface where a group of slices is
offset against the next group, at least two in the hemisphere and one toward the vermis.

Block-step measure (rigid GC fit of the mean of 4 slices before each boundary to the mean of the 4 after it):
the blocks are in the raw sections (sections mounted together at a common angle), not made by the registration.

| Block (slices) | Position | Raw (A20 coreg0nl) | A20 final | J20 final |
|---|---|---|---|---|
| 276-283 (TP3 Image 13-17) | 13.8-14.2 mm | -15 / +10 deg | -4 / +5 deg | -7 / +10 deg |
| 234-235 (TP2 Image 91) | 11.7 mm | +14..+19 deg | +1..+2 deg | 0..+1 deg |
| 322-326 (TP3 Image 38-39) | 16.1-16.3 mm | -9..-14 deg | -3..-4 deg | -3..-5 deg |
| 33-35 (next to interpolated slice 32) | 1.7 mm, vermis side | -3..-5 deg | -1 deg | -1..-2 deg |

Interpolated slices: only 32 (rank 11 of 391 boundaries) and 160 (rank 50) have a step nearby; 59, 120, 189, 228 do
not. TP changes (112, 251): 251 has a moderate step (0.51 mm at 253). The nearest-neighbour fits of every step
(jitter, STAGE1 neighbours, groupwise local templates) cannot turn a block back: inside a block all slices agree,
only its two edges show a mismatch. The jitter correction even made the far edge of 276-283 worse (+10 -> +16 deg),
and J's STAGE1 kept the block (12-14 deg) where A's reduced it (6-10 deg).

Pose-graph correction (`slice_posegraph.py`, `SLICEREG_JITTER_METHOD=posegraph`, output suffix `_pg`): rigid fits of
each slice to the slices at distances 1, 2, 3, 4, 6, 8, 12, 16; one robust least-squares solve for all poses
(rotations first, then translations); slow part removed (sigma 15); 2 outer passes. Offline test on the raw A20
stack (`posegraph_test.py`, both corrections applied the same way):

| Measure | none | jitter | pose graph |
|---|---|---|---|
| block step RMS median / 95th pct (mm) | 0.57 / 1.53 | 0.38 / 1.33 | **0.18 / 0.35** |
| boundaries with a block step > 0.5 mm | 217 | 137 | **7** |
| blocks 276-283 / 234 / 324 (deg) | -16,+10 / +19 / -15 | -13,+16 / +10 / -14 | **-0.4,+0.8 / -1.5 / -1.2** |
| blur-equalized NCC median / 5th pct | 0.475 / 0.114 | 0.824 / 0.466 | **0.841 / 0.581** |
| mask Dice median / 5th pct | 0.901 / 0.791 | 0.945 / 0.849 | **0.946 / 0.868** |
| slow-path change, max (mm) | - | 0.02 | 0.22 |

The block-step measure uses the same kind of rigid fit that the pose graph optimises (partly circular); the
through-plane views (folia continuous through the whole stack with rigid poses only) and the mask Dice are
independent and agree. Cost: about 3,000 rigid fits per pass (about 12 min per pass at rescale 20, 6 workers).
Full rescale-20 run P20 (`/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/zefir_sliceReg_optimized_v2_rescale_20_local_pg/`,
`SLICEREG_JITTER_METHOD=posegraph`, finished 2026-10-28, no errors), against A20 (no correction) and J20 (jitter):

| Measure (final `groupwise_iter9` unless stated) | A20 | J20 | P20 |
|---|---|---|---|
| block step RMS median / 95th pct (mm) | 0.147 / 0.467 | 0.143 / 0.485 | **0.128 / 0.290** |
| boundaries with a block step > 0.5 mm | 16 | 19 | **6** (all at 385-390, the small lateral end) |
| boundaries with a block rotation > 3 deg | 27 | 23 | **0** |
| blocks 276-283 / 234-235 / 322-326 (deg) | -4,+5 / +2 / -4 | -7,+10 / 0 / -5 | **-0.3,+0.7 / -1.3 / -0.7** |
| blur-equalized NCC median / 5th pct | 0.932 / 0.702 | 0.959 / 0.776 | **0.968 / 0.837** |
| mask Dice median / 5th pct | 0.974 / 0.925 | 0.979 / 0.934 | 0.980 / 0.933 |
| STAGE1 blur-eq NCC median / 5th pct | 0.856 / 0.511 | 0.926 / 0.622 | **0.950 / 0.783** |
| STAGE1 warp, total / non-affine RMS median (mm) | 0.58 / 0.25 | 0.29 / 0.21 | **0.24 / 0.18** |
| slow path vs A20: centroid max (mm) / angle max (deg) | - | 0.20 / 5.3 | 0.26 / 2.9 |
| count ratio registered / native, \|dev\| median / 95th pct | 0.026 / 0.066 | 0.020 / 0.074 | 0.019 / 0.073 |

- The blocks are gone at every level (coreg0nl, STAGE1, groupwise: no boundary above 3 deg). All other blocks
  keep a small common rotation of about -1 deg; this is the same at all of them and is the anatomy (the block fit
  of 4 + 4 slices sees the curve of the folia).
- Through-plane views (in-plane rows j = 87 and 108, sections 9-20 mm): A20 and J20 show a sharp break of the
  folia at about 14 mm; P20 shows them continuous.
- MRI registration of P20 (`mri_register.py`, from the reference): rotation vs reference 0 deg, SyN Dice
  0.853 -> 0.864, displacement median 0.83 mm, no folding. D99 labels: vermis cortex 25.5%, hemisphere cortex
  40.4% of the cells (J20: 25.1 / 40.4%).
- Cost: the pose-graph step took 21 minutes at rescale 20 (2 passes of about 10.5 minutes, 8 workers); the whole
  run took about 3 h 20 min (10:37-13:55). The later stages were not slower than in J20.

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
- Labels: `labels/synthetic_slices` = 1 on the interpolated slices (not measured); `labels/tissue` = 1 in tissue
  (since 2026-10-28; the definition of `tissue_mask.py`, see §3). If the D99 label files of the level exist
  (`atlas_labels.py`, §7.3), also `labels/d99_labels` and `labels/d99_regions` (same values as the NIfTI files;
  label names in the attributes under `d99_label_names`). `mri_register.py` makes the labels before its export.
  The `stack` export of the pipeline runs before `mri_register.py`, so it has no D99 labels; to add them, export
  it again: `uv run python macaque_CB/export_ome_zarr.py --run-dir <dir> --rescale <r> --level <level>
  --placement stack --out <dir>/<subject>_<level>_density_stack.ome.zarr --overwrite`.
- Our attributes are under `neuralarc.microscopy_scripts` in the root `zarr.json` (tck_view merges them there).
- Placement `stack`: the registered grid in mm, no anatomical claim.
  Placement `mri`: the same voxels (not resampled) with an NGFF 0.6 affine to D99 world (RAS, mm). Since
  2026-10-28 it is written by `mri_register.py` (§7) with the stored, checked affine (`--mri-affine`); the pipeline
  no longer makes it. A fresh fit for every export (the old behaviour, still the default of `export_ome_zarr.py`
  without `--mri-affine`) was not stable: the A20 export was turned by about 180 deg in-plane (anatomically wrong).
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

## 7. QC report and MRI registration (2026-10-28)

### 7.1 QC report (`qc_report.py`)

Runs at the end of every pipeline run (`SLICEREG_QC=1`) and can be run by hand:
`uv run python macaque_CB/qc_report.py --run-dir <output_dir> --rescale <r>`. It writes one self-contained page,
`<output_dir>/QC/index.html` (images embedded), plus `QC/metrics.json` and a cache of per-level measures
(`QC/cache/`, recomputed when a stack file is newer, or with `--recompute`). Nothing outside `QC/` is changed.

Content: run settings; continuity per level (adjacent NCC, blur-equalized NCC, tissue-mask Dice: median and 5th
pct); continuity along the stack and images of the weakest adjacent pairs; warp per slice (STAGE1 map) and per
groupwise step; per-slice count ratio (registered / native; independent of the registration's cost); slow path
and jitter poses; through-plane views at physical aspect and a slice montage; MRI registration results if
`mri_register.py` has run. Stacks finer than 0.2 mm are block-summed in-plane to about 0.2 mm for the QC (stated
in the page), so it stays fast at rescale 5 and the numbers compare between rescales.

### 7.2 MRI registration (`mri_register.py`)

Point it at any run and level (the level to use may not be the last groupwise iteration):
```bash
uv run python macaque_CB/mri_register.py --run-dir <output_dir> --rescale <r> [--level <level>]
```
1. Affine (shear-free): from the stored reference (`mri_reference/zefir_stack_to_D99_reference.json` + `.mat`),
   refined; or `--affine FILE` as it is; or a full search (`--no-reference`) that also tries the in-plane 180 deg
   alternative.
2. Stability check against the reference (rotation <= 10 deg, scales within 25%; `--max-rotation-deg`,
   `--max-scale-change`): on failure it stops with exit 2 and writes nothing but a FAILED entry in the QC.
3. Nonlinear (unless `--no-nonlinear`): SyN (MI, flow sigma 3, gradient step 0.1, iterations 100/70/30) inside the
   region that the stack covers. D99 is another animal: this is atlas mapping, not part of the reconstruction.

Outputs in `<output_dir>/mri_registration/<level>/` (README.md there explains frames and use): the transforms
(`stack_to_D99_affine.mat`, `stack_to_D99_warp.nii.gz`, `stack_to_D99_inverse_warp.nii.gz`), densities on the D99
grid (`density_D99grid_affine.nii.gz`, `density_D99grid_nonlinear.nii.gz`, cells/mm^3), and the OME-Zarr
`<subject>_<level>_density_mri.ome.zarr`. QC in `QC/mri/<level>/`, shown in `QC/index.html`.

Orientation: the cerebellum outline (one hemisphere + vermis) is nearly symmetric under an in-plane turn of
180 deg, so Dice cannot choose the orientation (J20: 0.838 vs 0.839, and the higher one was anatomically wrong:
brainstem dorsal). Mattes MI on the intensities separates them clearly (-0.09 to -0.12 for the correct placement in
J20, J40, A40; -0.056 to -0.059 for the turned one, which the old A20 export had). The full search chooses by MI.

Reference: made from J20 (`..._rescale_20_local_jit`, `groupwise_iter9`) with a full search, checked by eye
(folia dorsal and posterior, brainstem and peduncles ventral and anterior, vermis medial), saved with
`--save-reference`. Scales MRI -> stack 0.68 / 0.77 / 1.00 (in-plane shrinkage of the sections relative to D99,
about 25-30%, is real tissue processing plus fit uncertainty).

Results (affine from the reference, then SyN; Dice in the SyN region, which differs from the affine table's Dice):
- J20: Dice 0.853 -> 0.860, MI -0.099 -> -0.199, displacement median 0.72 mm (95th pct 1.47, max 2.35),
  Jacobian 5th-95th pct 0.70-1.55, no folding.
- J40: Dice 0.856 -> 0.851 (Dice and MI disagree here), MI -0.111 -> -0.222, displacement median 0.81 mm,
  Jacobian 0.68-1.60, no folding.
- The SyN is not gentle: local volume changes up to about +/-50% (5th-95th pct). This is expected for a different
  animal; use the affine outputs when the stack's own shape matters.
- Tests: the evidence that the check works is a 180 deg turned affine given with `--affine`, which fails
  (176 deg, exit 2, nothing written). A run from the reference passes trivially (the refinement kept it: 0 deg).
- The full search is not fully repeatable (the 2D orientation search has no fixed seed): two J20 searches
  differed by 4 deg. The stored reference removes this for later runs.
- Old `mri` exports checked against the reference (2026-10-28): A20 was turned (177 deg) and is replaced by a
  `mri_register.py` export; all others agree within 8 deg, also
  `ome_zarr_exports/zefir_r5_gw9_mri.ome.zarr` (7.7 deg).

QC report at rescale 5 (test with the rescale-5 stack, 2026-10-28): about 5 s per level and 2.1 GB peak memory
for the stack measures; the map warps (about 400 maps per level) add an estimated 10-20 minutes.

### 7.3 D99 labels in the stack (`atlas_labels.py`, 2026-10-28) - WORK IN PROGRESS

Status: first version by atlas propagation, not validated. Do not use the lobule or vermis labels for analysis
yet. Open decisions and the plan for the next step: `LOBULE_LABELS_HANDOFF.md`.

`mri_register.py` runs it at its end (`--no-labels` turns it off); for another level or other settings:
```bash
uv run python macaque_CB/atlas_labels.py --run-dir <output_dir> --rescale <r> [--level <level>] \
    [--anterior-vermis-halfwidth-mm 4.0] [--fill-mm 1.0]
```
- Source: D99 v2.0 labels (`local_extras/cerebellum_target/D99_cerebellum_brainstem_labels_crop.nii.gz`, the grid
  of the fitted template). Names from `D99_v2.0_labels_semicolon.txt` of the AFNI D99 v2.0 package, copied for the
  79 values in the crop to `mri_reference/D99_v2.0_labels_cerebellum_brainstem.tsv`.
- Transforms: affine (inverted) + inverse SyN warp, nearest label (ANTs genericLabel), onto the full stack grid in
  slabs of 32 sections. Check of the direction on J20: the D99 mask pulled back this way has Dice 0.873 with the
  stack tissue (affine only 0.818; the transforms in the wrong order 0.804).
- D99 labels grey matter structures only. Tissue in the D99 mask without a label is split with the D99 region map
  (`D99_cerebellum_brainstem_regions_crop`, 1 cerebellum, 2 brainstem): 252 = cerebellum (white matter in D99),
  253 = brainstem (e.g. the reticular formation, which D99 does not label). In a first version both were one code
  (254, "mostly white matter"); a check showed that 80% of its cells were in the D99 brainstem, at cortex-like
  density. Tissue outside the D99 mask (255) is mostly brainstem cut by the D99 crop and a rim where the shapes
  differ (median depth 0.9 mm); rim tissue within 1 mm of a label takes the nearest label (`--fill-mm`).
- Vermis: D99 has no vermis label. Lobules VI-X (113-117) are vermal (the hemispheric parts have their own labels:
  simple lobule, crus I/II, paramedian lobule, paraflocculus, flocculus). Lobules I-V (108-112) run from the midline
  to the lateral edge; here they are vermis where |D99 x| <= 4.0 mm (the median half-width of D99 lobule VI;
  VI-X together: median 2.5 mm, 90th percentile 4.25 mm). This cut-off is a choice; the x map allows another one.
- Left/right is that of the D99 placement (the reference puts the sections on D99's right). Section 75 is at the
  D99 midline in J20.

J20 (`groupwise_iter9`), cells in measured sections: vermis cortex 25.1%, hemisphere cortex 40.4%, deep nuclei
0.7%, other D99 labels (brainstem, fibre bundles) 9.6%, D99 cerebellum without a label 3.5%, D99 brainstem without
a label 13.6%, outside the D99 mask 7.2% (14.6% before the 1 mm rim fill). J40 (full `mri_register.py` run, which
also checks the integration): 24.9 / 40.3 / 0.7 / 9.8 / 3.5 / 13.8 / 6.9%. About 1 minute, 0.76 GB peak at
rescale 20; rescale 5 not tested yet (estimated peak about 6 GB).

Open point: in D99 coordinates the stack ends at x of about 16 mm (sections 375-397: 105 -> 38 mm^2 of cerebellar
label area), while 75% of D99 crus I and crus II lies lateral to 16 mm. So crus I/II get few cells (3 M each) and
the posterior paramedian lobule many (80 M). Either the sections stop before the lateral edge, or the fit
(free scales; it kept the section axis at scale 0.995 (row norm of the affine) while the in-plane axes are 0.77 and
0.68) puts the lateral end too far medially. The lateral labels depend on which is true, and so does the vermis
width: with a section scale of about 0.72, the 4 mm cut in D99 x would be about 2.9 mm in the stack (about 20
sections narrower on the covered side).

Accuracy: D99 is another animal at 0.25 mm and the SyN is smooth, so lobule boundaries can be off by about one
folium (most likely at the primary fissure and in lobules I-V). Check the QC images (sagittal sections with label
edges and names; views across the sections that show the vermis band).

### 7.4 Count-preserving stack (`count_preserving_stack.py`, 2026-10-29)

Why: the pipeline stacks pull the native counts through the warps without a Jacobian. Each final pixel holds the
native count per native pixel at that point (a density), which is correct for images and masks. But the total of a
section changes with the area change of the warp: QC "Counts", P20 `groupwise_iter9`, median 1.9%, up to 21% (end
of the stack, sections 380-397).

Checks of the input, 2026-10-29: the first step sums the counts in blocks (`downsample_image`: `block_reduce` with
`numpy.sum`). The input TIFFs hold whole cell counts per 10 µm pixel (values 0-5). The native section has the same
total as its TIFF (section 100: 2,040,290 at rescale 20 and at rescale 5, HPC), and no negative values. The pad
value of `downsample_image` is `-sdf_clip_val`, and `sdf_clip_val` is 0 because the scripts set
`use_signed_distance_weighting_for_registration = False` (so the cost function is MutualInformation and the
missing-slice method is `intermediate_nonlin_mean`); the -10/-50/-100 values apply only to the SDF mode.

Method: each native section goes through the same chain of pull maps as the pipeline (coreg0nl -> rigsyn_iter4_win12
-> groupwise_iter3..N; sequential, because the composition helpers are marked broken). Each step multiplies the
pulled values by |det J| of its map (J = derivative of the source voxel coordinates with respect to the output pixel
index, central differences; 0 where the map samples outside the source canvas). An image of ones on the native grid
goes through the same steps and gives the native area per final pixel.

Outputs beside the stack (new files; the pipeline stacks and csv files are not changed):
`<subject>_<level>_counts_stack.nii.gz` (cells per final pixel), `_native_area_stack.nii.gz` (native pixels per
final pixel), `_counts_per_section.tsv`, `_counts_stack.json`, `_counts_stack_README.md`. Density of a region R in
cells/mm^3 = sum_R(counts) / (sum_R(native area) x native pixel area x 0.05 mm). The interpolated sections are 0 in
both stacks and left out of all totals.

How to use the stacks (added 2026-10-29 after a question from the user; `usage_notes()` in
`count_preserving_stack.py` writes these rules with the run's numbers into the json key `usage` and into
`_counts_stack_README.md`; the QC page "Counts" and `QC/METHODS.md` §9 show them):

| Need | Use | Factor (cells/mm^3) |
|---|---|---|
| cells in a region | sum of the counts stack over the region | - |
| density of a region | factor x sum(counts) / sum(native area) | 1 / (native pixel area mm^2 x 0.05 mm): 8000 at rescale 5 (50 um), 500 at rescale 20 (200 um) |
| density image (per voxel) | factor x pipeline stack (`<subject>_<level>_stack.nii.gz`) | same |

Divide the counts by the area; do not multiply them. Do not divide the two count-preserving stacks voxel by voxel:
the native area is near 0 at the tissue and canvas edges, so that ratio is noisy there. Inside the tissue it equals
the pipeline stack (P20, voxels with native area > 0.5 and stack > 2: median ratio 1.0001, 5th-95th percentile
0.969-1.025; over all of them, sum(counts)/sum(area) 193.8 against the area-weighted stack mean 193.7).
`count_preserving_stack.py --run-dir <run> --notes-only` adds the rules and the README to an existing count stack
(no recomputation); done for P20 and the HPC rescale-5 run on 2026-10-29.

P20 results: count-preserving total / native total per section, |deviation| median 0.05%, 95th percentile 0.13%,
max 0.25% (pipeline stack: median 1.9%, max 21%); no counts on pixels with det J < 0; counts / native area
reproduces the pipeline stack (NCC 0.9999, total ratio 1.0008); native area per final tissue pixel 0.74-1.26
(5th-95th percentile). 5 s with 6 workers at rescale 20.

Users: `atlas_labels.py` adds the columns `cells_cp`, `native_volume_mm3` and `density_cp_cells_per_mm3` to
`_D99_labels.tsv` when the stack exists (the first columns are unchanged). P20: per label, `cells_cp` differs from
`cells` by 1-3%. The QC report adds the count-preserving ratio to "Counts" and plots the cells per section split by
the D99 regions. In P20, most of the decrease of the native totals after section 120 is in the brainstem part.
`find_levels` (QC, MRI, labels) now matches the groupwise levels exactly, so the new `_counts_stack` and
`_native_area_stack` files are not taken as levels. `cleanup_run.py` keeps them (run-level files).

### 7.5 Methods text of a run (`methods_report.py`, 2026-10-29)

`qc_report.py` writes `QC/METHODS.md` (scientific English) and shows it as the last section of the QC page; a
failure is caught and reported in the page. By hand: `methods_report.py --run-dir <run> [--check]`.

Sources, in this order of trust (each parameter in Appendix A names its source):
- run: `run_settings.json`. Since 2026-10-29 the run scripts add keys during the run (`_update_run_settings`;
  existing keys are kept): `git_dirty`, `packages` (ants, numpy, scipy, nibabel, skimage, sklearn, pandas),
  `n_sections`, `input_files`, `pipeline` (SDF mode, cost function, missing-section method, template type,
  `use_resolution_in_registration`, neighbour targets, MRI anchor), `scaling_factor` and `canvas_shape` (after they
  are computed), `sharpening` (the dataset multipliers), `cascade`, `stage1`, `groupwise`.
- log: every `antsRegistration` argument list in the run log, grouped by level (transform, metric, bins, sampling,
  iterations, shrink factors, sigmas, number of metric terms and their weights); pose-graph lines; scaling factor;
  skip, fallback and exit messages. The summary is cached in `QC/cache/methods_log.json` (P20 log 245 MB: 5 s).
- script: the copy of the main script in the run folder, parsed with `ast` (offsets, Gaussian sigmas, groupwise
  keywords, which stacking calls get the sharpening multipliers).
- code: 31 statements about function internals, each a regular expression against the file at the run's git
  commit (`git show`), else the working tree (named in Appendix B). A failed check marks the statement
  UNVERIFIED in the text; `--check` exits 1.
- files: QC/metrics.json, QC/mri/*/qc.json, atlas_labels.json, `_counts_stack.json`, the jitter poses csv, and
  (when it exists) `_granular_mask.json`.

References: checked on 2026-10-29 against the publisher, PubMed Central or Project Euclid records (authors, title,
journal, volume, pages, DOI): Avants 2008 (SyN), Avants 2011 (ANTs metrics), Tustison 2021 (ANTsX/ANTsPy),
Huntenburg 2018 (nighres: the registration wrapper is a modified copy of its `embedded_antspy_2d_multi`), Maes 1997 (MI), Mattes 2003 (Mattes MI), Huber 1964,
Ashburner & Friston 2000 (Jacobian modulation: p. 815 "multiplies the partitioned images by the relative voxel
volumes ... the Jacobian determinants of the deformation field"), Reveley 2017 (D99), Saleem 2021 (D99 v2.0).

Facts found while writing it (all checked in the code or the log):
- `coreg0nl` has no optimisation: `do_reg(run_rigid=False, run_syn=False)` calls antsRegistration with
  `Rigid[0.1]` and `--convergence [0,1.0,2]`, after `ignore_orient=True` has replaced the headers by centred grids.
  So coreg0nl is a centre alignment on the canvas of the largest section; the rigid alignment comes from the pose
  graph and from STAGE1.
- The smoothing sigmas of STAGE1 and groupwise (`16x8x4x2x1`, no suffix) are voxels: ANTs reads a sigma without a
  `mm`/`vox` suffix as voxels (`antsRegistrationTemplateHeader.h`). (A first reading in this session assumed
  physical units, i.e. micrometres; that was wrong.)
- STAGE1 metric weights: Gaussian of the offset, the stack template gets the weight of offset 0, and
  `embedded_antspy_2d_multi` divides all weights by their sum (logged interior weights: 0.116, 0.116, 0.038 ...).
- The sharpening multipliers are passed only to the `coreg0nl` stacking; the later levels interpolate the
  excluded sections without the unsharp mask.
- With a stored reference, `mri_register.py` always uses "refined from reference"; it does not choose between the
  candidates (the HPC run: refined Dice 0.786 vs reference 0.805, rotation 4.6 deg from the reference).
- P20 ran with `slice_posegraph.py` before it was committed (commit feda426 does not contain it); its checks use
  the working tree, and Appendix B says so.
- The SDF mode is off in both scripts, so `sdf_clip_val` = 0, the pad value of the downsampling is 0, the cost
  function is MutualInformation and the missing-section method is `intermediate_nonlin_mean`.

