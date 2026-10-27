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
| `SLICEREG_MRI_DIRECTION` | 1 | D99 hemisphere the stack is fitted to (+1 right, -1 left); fixed so outputs are comparable |
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
| A | 40 | `coreg0nl` (original) | running |
| B | 40 | `previous` | running |
| C | 40 | `previous` + MRI anchor | running |

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

### 4.3 Missing-slice interpolation check (rescale-5 `groupwise_iter9` stack)

| slice | NCC prev-next | NCC synthetic-neighbours | note |
|---|---|---|---|
| 59, 189 | 0.91-0.94 | ~0.97 | good |
| 120, 228 | 0.69-0.82 | 0.87-0.94 | acceptable |
| 160 | 0.61 | ~0.85 | neighbours disagree |
| 32 | 0.41 | ~0.76 | neighbours disagree strongly (pair 33-34 is also among the worst in the stack) |

Synthetic slices keep 80-93% of the edge sharpness of their neighbours. Where they are weak, the cause is
poor alignment of slices i-1 and i+1 across the 100 um gap, not the interpolation itself.

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
- Placement `stack`: the registered grid in mm, no anatomical claim.
  Placement `mri`: the same voxels (not resampled) with an NGFF 0.6 affine to D99 world (RAS, mm), from a fresh
  shear-free fit of the stack to D99 (`--direction`, default +1 = right hemisphere).
- Open in tck_view: `uv run --project ~/Documents/code/tck_view arc <file>.ome.zarr`.

Exports of the existing rescale-5 `groupwise_iter9` stack (made before this branch):
`/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/ome_zarr_exports/zefir_r5_gw9_{stack,mri}.ome.zarr`
