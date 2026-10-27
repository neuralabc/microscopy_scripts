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

Next: resample D99 into the stack grid (one MRI plane per slice) and use it in STAGE1 in place of the
whole-stack median target. This adds new files but must not change the names in §3.
