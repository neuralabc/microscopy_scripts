# Scripts and tools (branch `enh-local_uv_mri_anchor`)

Short guide to the scripts. Details, results and decisions: `REGISTRATION_NOTES_local.md` (cited as §n).
Run every command from the repo root through uv: `uv run python macaque_CB/<script> ...`.

## 1. Setup

```bash
cd <repo> && uv sync            # Python 3.12, antspyx 0.6.3 (pyproject.toml, uv.lock)
```
- `nighres` is not needed: `env_compat.py` provides the parts that the pipeline uses (§1).
- D99 files (not in git): `macaque_CB/local_extras/cerebellum_target/` on the workstation; on the HPC the
  `_hpc` script uses `/data/neuralabc/johmat/phase_ml/D99/D99_v2.0_dist/cerebellum_target`. `SLICEREG_D99_DIR`
  sets another folder (for all scripts). Without the files, the main script skips the MRI registration and labels.
- tck_view (optional, for the OME-Zarr export): a checkout at `~/Documents/code/tck_view`, or set
  `SLICEREG_TCK_VIEW`. Without it, the export is skipped.

## 2. Main pipeline

| Script | Machine | Default input | Default output |
|---|---|---|---|
| `run_slice_registration_optimized_newReg_sdf_v2.py` | original, do not edit | - | - |
| `run_slice_registration_optimized_newReg_sdf_v2_local.py` | workstation (8 workers) | `/mnt/ssd/.../all_TP_image_idxs_file_lookup.csv` | `/mnt/ssd/.../zefir_sliceReg_optimized_v2_rescale_{r}_local_pg/` |
| `run_slice_registration_optimized_newReg_sdf_v2_hpc.py` | HPC (96 workers) | `/data/neuralabc/neuralabc_volunteers/macaque/all_TP_image_idxs_file_lookup.csv` | `/tmp/zefir_sliceReg_optimized_v2_rescale_{r}_hpc_pg/` |

The `_local` and `_hpc` scripts are the same pipeline. Only the configuration lines differ (`diff` the two files).
Steps: downsampling, coreg0nl + pose-graph correction, STAGE1 (rigid + SyN), groupwise (10 iterations), tissue
mask, count-preserving stack, OME-Zarr (`stack` placement), MRI registration + D99 labels, QC report.

HPC, full resolution (rescale 5 = 50 µm in-plane, the default):
```bash
mkdir -p /tmp/zefir_work && cd /tmp/zefir_work          # ANTs can leave files in the working directory
SLICEREG_SMOKE=1 uv run --project <repo> python <repo>/macaque_CB/run_slice_registration_optimized_newReg_sdf_v2_hpc.py   # 3-min environment test
nohup uv run --project <repo> python <repo>/macaque_CB/run_slice_registration_optimized_newReg_sdf_v2_hpc.py > run.log 2>&1 &
```
- The image paths in the csv are used as they are (as in the original script). The script stops at the start
  if an input image is missing, and warns if the D99 files are missing.
- Other inputs: `SLICEREG_LOOKUP_CSV`, `SLICEREG_OUTPUT_ROOT` (default `/tmp/`), `SLICEREG_D99_DIR`,
  `SLICEREG_MAX_WORKERS` (96).
- Disk: P20 (rescale 20) is 6.1 GB in about 88,000 files. Rescale 5 has 16 times the in-plane voxels: expect
  about 100 GB in `/tmp`.
- `/tmp` may be cleared: copy the output folder off when the run ends.
- Rescale 5 with this version has not been run yet. Expected: the pose graph takes about 15 min per pass (2
  passes), and the QC takes 10-20 min. The original script used 50 workers at rescale 5 on this HPC.

All options are `SLICEREG_*` environment variables (§2): e.g. `SLICEREG_RESCALE`, `SLICEREG_JITTER_METHOD`
(posegraph | jitter), `SLICEREG_GROUPWISE_ITERS`, `SLICEREG_QC`, `SLICEREG_MRI_REGISTER`, `SLICEREG_EXPORT_ZARR`,
`SLICEREG_COUNTS_STACK`.
Each run writes `run_settings.json`. Output files and the transform chain: §3 (Mat's processing depends on them).

## 3. Tools for a finished run (point them at a run folder and a level)

| Script | Does | Command |
|---|---|---|
| `qc_report.py` | QC page `<run>/QC/index.html`: continuity, warps, counts, images per level; MRI and label parts if present | `qc_report.py --run-dir <run> --rescale <r>` |
| `mri_register.py` | Stack to D99: stored reference affine (+ stability check, exit 2 on failure), SyN, densities on the D99 grid, `mri` OME-Zarr; then runs `atlas_labels.py` | `mri_register.py --run-dir <run> --rescale <r> [--level <level>]` |
| `atlas_labels.py` | D99 labels in the stack grid: `_D99_labels`, `_D99_regions` (vermis / hemisphere / nuclei), `_D99_x_mm`, cell table. **Work in progress** (`LOBULE_LABELS_HANDOFF.md`) | `atlas_labels.py --run-dir <run> --rescale <r> [--level <level>]` |
| `methods_report.py` | Methods text of a run in scientific English, `<run>/QC/METHODS.md`, written from the run's own records (settings, log, script copy, sidecars); statements about code internals are checked against the source at the run's commit (Appendix B); verified references. `qc_report.py` runs it and shows it in the QC page | `methods_report.py --run-dir <run> [--check]` |
| `count_preserving_stack.py` | Count-preserving stack of a level: the native counts through the same chain of maps, each step times \|det J\|, so that section totals keep the native totals (P20: deviation median 0.05%, max 0.25%; pipeline stack up to 21%). Also writes the native area per final pixel, a per-section table, a json summary and `_counts_stack_README.md` (how to use the stacks, with the factor to cells/mm^3: 8000 at rescale 5). Use it for cell totals and region densities (factor x sum(counts) / sum(native area)); for a density image use factor x the pipeline stack | `count_preserving_stack.py --run-dir <run> [--final-level groupwise_iter9] [--workers 8]`; `--notes-only` adds the README and the json rules to an existing count stack |
| `tissue_mask.py` | Tissue mask of a level (counts > 2 per 50 µm voxel, holes filled) | `tissue_mask.py --run-dir <run> --rescale <r> --level <level>` |
| `cleanup_run.py` | Removes intermediate files; keeps the native sections and the transform chain that pushes data (e.g. Purkinje-cell maps) into the final space. Dry run by default; `--verify N` proves that the kept chain reproduces the final sections; `--execute` removes (P20: 4.0 of 6.3 GB, 76,000 files) | `cleanup_run.py --run-dir <run> --verify 5 [--execute] [--drop-inverse]` |
| `export_ome_zarr.py` | OME-Zarr (NGFF 0.6) of a level in cells/mm³, `stack` or `mri` placement, with label maps | `export_ome_zarr.py --run-dir <run> --rescale <r> --level <level> --placement stack --out <file>.ome.zarr` |

`--level` defaults to the last groupwise iteration. An earlier level is allowed (e.g. if the later warps are too
strong). Reference data in `mri_reference/`: the stack-to-D99 reference affine and the D99 label names.

## 4. Comparison and test tools (offline, they do not change a run)

| Script | Does |
|---|---|
| `evaluate_registration.py --rescale <r> --out <dir> [--no-mri] [--reference A] A=<run> B=<run>` | Compares runs: blur-equalized NCC, mask Dice, slow path, STAGE1 warp size, MRI agreement |
| `posegraph_test.py <run> <r> <out>` | Pose graph against jitter correction on a raw coreg0nl stack (block steps, NCC, Dice) |
| `block_steps.py --run-dir <run> --rescale <r> [--level <level>] [--lookup-csv <csv>]` | Steps between groups of 4 sections (blocks mounted at a common angle, §4.8) |
| `count_ratio.py --levels <l1,l2> A=<run> [B=<run>]` | Cell count per section after the warps against the native section |
| `jitter_correction_test.py <stack> <res_mm> <out>` | Jitter correction alone on a stack |
| `mri_pose_consistency.py`, `drift_control_test.py` | Tests of MRI-based and MRI-free drift control (§4.4, void; §4.5) |

## 5. Modules (imported, not run)

| Module | Contents |
|---|---|
| `env_compat.py` | Import first: 1 thread per worker, OOM priority, nighres and ANTsPy shims |
| `slice_posegraph.py` | Pose-graph rigid correction (pairs at 1-16 slices, robust solve), the default |
| `slice_jitter.py` | Nearest-neighbour jitter correction; `correct_level_in_place` writes the poses into coreg0nl |
| `slice_registration_extensions.py` | Parallel coregistration with neighbour options; groupwise v2 (file names correct past iteration 10) |
| `mri_anchor.py` | D99 loading, stack-to-D99 affine fit, tissue threshold |
| `_tckview_write_ome_zarr.py` | Runs in tck_view's environment and writes the OME-Zarr |
| `slice_registration_functions.py`, `embedded_antspy_smthMod.py` | Original library (unchanged) |

## 6. Other files (not changed on this branch)

- QuPath scripts (`*.groovy`, `purk_classifier.json`): cell detection, counts and Purkinje-cell classification
  on the histology images; they make the count maps that the pipeline reads.
- `notebooks/*Apply_Mappings*`: push other data through the transform chain into the registered space (§3).
- `legacy/`: older versions of the registration scripts.

## 7. Push data into the final space, and clean up a run

### 7.1 Files needed to push data (e.g. Mat's Purkinje-cell masks) into the final space

The push applies, per section, a chain of forward maps (`_ants-map.nii.gz`, pull-style coordinate maps) to an
image on the native section grid. The chain to the final level `groupwise_iterN` (from
`slice_registration_functions.get_transform_chain_for_level`, which reads `<subject>_forward_maps.csv`):

| Step | File per section (`<base>` = `zefir_NNNN_<image>_downsample_10p002um_pix`) |
|---|---|
| source grid | `<base>.nii.gz` (native section; a mask is first downsampled to this grid) |
| 1. coreg0nl | `init_translation_slice_NNNN/<base>_coreg0nl_ants-map.nii.gz` |
| 2. STAGE1 | `<base>_coreg12nl_win12_rigsyn_4_ants-map.nii.gz` |
| 3.-9. groupwise | `<base>_coreg12nl_win12_rigsyn_4_groupwise_iter{3..N}_ants-map.nii.gz` |
| check | `<base>_coreg12nl_win12_rigsyn_4_groupwise_iterN_ants-def0.nii.gz` (the counts pushed the same way) |

Apply the maps in this order, each with `apply_coordinate_mapping_2d` (as in
`notebooks/2026_04_Apply_Mappings_python_CLEAN.ipynb`). The inverse maps (`_ants-invmap`) of the same levels
bring final-space data (e.g. the D99 labels) back to the sections, in the reverse order. Interpolated sections
(32, 59, 120, 160, 189, 228) have maps, but their final `def0` is interpolated from the neighbours: leave them out
or treat them in the same way. Everything else in a run folder is not needed for a push.

### 7.2 Cleanup (`cleanup_run.py`)

```bash
uv run python macaque_CB/cleanup_run.py --run-dir <run> --verify 5             # 1. report + check (removes nothing)
uv run python macaque_CB/cleanup_run.py --run-dir <run> --verify 5 --execute   # 2. remove
```
1. Read the report: GB and files per category, "keep" and "remove". "keep: unknown per-section file" must be 0 or
   understood (unknown files are never removed).
2. `--verify 5` must say `verify: OK` (NCC >= 0.95; it is 1.000 on P20). It pushes 5 sections through the kept
   maps only and compares them with the final `def0`.
3. Then run with `--execute`. The script refuses if a native section or a chain map is missing. It writes every
   removed path to `<run>/cleanup_manifest.txt`.

| Kept | Removed |
|---|---|
| native sections; chain forward maps; chain inverse maps (`--drop-inverse` removes them); final `def0`; stacks, templates, csv / json / log files, tissue mask, D99 labels; `QC/`, `mri_registration/`, `missing_slices/`, `*.ome.zarr` | maps of levels not in the chain (cascade_0, STAGE1 levels 0-3, win1 / win2 / plain coreg12nl, coreg1nl / coreg2nl, groupwise_iter0-2); deformed sections of non-final levels (`def0`-`def10`); per-section templates; ANTs intermediate files |

Sizes: P20 (rescale 20) 6.3 GB -> 2.3 GB (76,000 files removed). Rescale 5 (estimate, x16): about 100 GB ->
about 36 GB, or about 25 GB with `--drop-inverse`.

After a cleanup:
- Only the final level and the chain remain. To use an earlier groupwise iteration as the final level, decide
  before the cleanup and pass `--final-level groupwise_iterK`: the chain then ends at K, and the maps of later
  iterations are removed.
- The maps csv files are not changed; their columns for removed levels point to absent files.
- The QC report keeps its cached measures but cannot measure removed levels again; `mri_register.py`,
  `atlas_labels.py` and `export_ome_zarr.py` still work on the final level (they use its stack).
- The removal cannot be undone. Copy a run first if its intermediate levels may be needed (e.g. to compare
  registration variants).
