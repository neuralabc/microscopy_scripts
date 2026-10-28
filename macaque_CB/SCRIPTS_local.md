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
mask, OME-Zarr (`stack` placement), MRI registration + D99 labels, QC report.

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
(posegraph | jitter), `SLICEREG_GROUPWISE_ITERS`, `SLICEREG_QC`, `SLICEREG_MRI_REGISTER`, `SLICEREG_EXPORT_ZARR`.
Each run writes `run_settings.json`. Output files and the transform chain: §3 (Mat's processing depends on them).

## 3. Tools for a finished run (point them at a run folder and a level)

| Script | Does | Command |
|---|---|---|
| `qc_report.py` | QC page `<run>/QC/index.html`: continuity, warps, counts, images per level; MRI and label parts if present | `qc_report.py --run-dir <run> --rescale <r>` |
| `mri_register.py` | Stack to D99: stored reference affine (+ stability check, exit 2 on failure), SyN, densities on the D99 grid, `mri` OME-Zarr; then runs `atlas_labels.py` | `mri_register.py --run-dir <run> --rescale <r> [--level <level>]` |
| `atlas_labels.py` | D99 labels in the stack grid: `_D99_labels`, `_D99_regions` (vermis / hemisphere / nuclei), `_D99_x_mm`, cell table. **Work in progress** (`LOBULE_LABELS_HANDOFF.md`) | `atlas_labels.py --run-dir <run> --rescale <r> [--level <level>]` |
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
