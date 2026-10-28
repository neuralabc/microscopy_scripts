# Handoff: lobule and vermis labels for the zefir cerebellum stack (WORK IN PROGRESS)

Written 2026-10-28, branch `enh-local_uv_mri_anchor`. Read `SCRIPTS_local.md` (tools) and
`REGISTRATION_NOTES_local.md` §3 (output contract), §7.2 (MRI registration), §7.3 (current labels) first.

## 1. Goal

Each tissue voxel of the registered stack gets a cerebellar lobule label (I-X and the hemispheric lobules) and a
vermis / hemisphere label. The labels must be defensible for cell counts per lobule. They must be on the stack
grid of a chosen registration level, so that Mat's data (pushed into the same space, §3) can use them.

## 2. Status: what exists

- `atlas_labels.py`: atlas propagation from D99 v2.0. The stack-to-D99 transforms of `mri_register.py` (affine
  from a checked reference + smooth SyN) pull the D99 labels back to the stack, nearest label. Outputs beside the
  stack: `{subject}_{level}_D99_labels.nii.gz`, `_D99_regions.nii.gz`, `_D99_x_mm.nii.gz`, `_D99_labels.tsv`.
  QC images are in `QC/index.html` (MRI part).
- `mri_register.py` and the main scripts run it by default.
- Results exist for J20, J40 and P20 (rescale 20 / 40, `groupwise_iter9`), under
  `/mnt/ssd/Macaque_CB/processing/results_from_cell_counts/zefir_sliceReg_optimized_v2_rescale_{20,40}_local_{jit,pg}/`.
  P20 (`_rescale_20_local_pg`) is the current default pipeline. Vermis cortex is 25%, hemisphere cortex 40% of
  the cells (stable across J20, J40, P20).
- Label names: `mri_reference/D99_v2.0_labels_cerebellum_brainstem.tsv` (from the AFNI D99 v2.0 package).

## 3. Known problems (why this is not done)

1. **Different animal, coarse atlas.** D99 is 0.25 mm. The SyN is smooth (flow sigma 3, median displacement
   0.8 mm). It aligns the outline and the coarse intensity, not the fissures. Lobule boundaries can be off by
   about one folium. Nothing has measured this yet.
2. **No vermis label in D99 for lobules I-V.** VI-X have vermal labels. I-V run from the midline to the lateral
   edge. The current rule (vermis if |D99 x| <= 4.0 mm) is a guess from the width of D99 lobule VI.
3. **Lateral coverage.** In D99 coordinates the stack ends at x of about 16 mm. 75% of D99 crus I/II lies lateral
   to that, so crus I/II get few cells and the paramedian lobule many. The affine keeps the section axis at scale
   0.995 but shrinks the in-plane axes to 0.68-0.77. Either the sections stop before the lateral edge, or the
   section-axis scale is wrong. This also moves the vermis boundary by about 1 mm (about 20 sections). Ask the
   user whether the last sections (about 390-397) show the lateral tip.
4. **Unlabelled tissue.** D99 leaves white matter (code 252) and much of the brainstem (253) without a label. The
   D99 crop cuts the brainstem (255).

## 4. Decision needed first (discuss with the user before you build)

| Option | Idea | For | Against |
|---|---|---|---|
| A. Better propagation | Keep D99. Improve the fit: section-axis scale, a less smooth SyN inside the cerebellum, a 2D refinement per section on the folial pattern | Little new code; full label set | Still another animal's boundaries; validation is hard |
| B. Direct anatomy | Labels only from zefir: find the fissures in the cell density, split the white-matter tree into branches, give each branch and its cortex a lobule | True boundaries of this brain; no atlas bias | More work; needs a seed per lobule (by hand or from D99); the vermis edge in I-V is subtle |
| C. Hybrid (recommended to discuss first) | D99 gives the initial label and the names. The boundaries then move to the nearest fissure in zefir's own data (e.g. watershed or graph cut on the cortex, with D99 labels as seeds and fissures as edges) | Names from the atlas, boundaries from the data | Needs a good fissure map; seeds can be wrong where D99 is off by a folium |

Other decisions: the vermis definition (paravermal sulcus where visible, a fixed width in zefir mm, or the
deep nuclei as a guide); the nomenclature (Larsell I-X + HVI, HVIIA, ... or the D99 names); which registration
level (the last groupwise iteration or an earlier one); and whether only cerebellar cortex gets lobule labels or
the white matter too.

## 5. Data that can help a direct or hybrid approach

- The stack: cell counts per voxel (densities about 110-150k cells/mm³ in cortex). The granular layer is dense;
  the molecular layer and the white matter are sparse. So fissures and the white-matter tree can be seen. Use
  rescale 20 for development and rescale 5 (50 µm) for the final result.
- Section continuity is now good enough for 3D work: the pose graph removed the blocks of sections at a common
  angle (§4.8), and the folia are continuous across the stack.
- Purkinje cells: `compute_PC_count_csv.groovy` runs a QuPath Purkinje classifier (`purk_classifier.json`). If
  per-section Purkinje maps exist, they trace the cortex layer of each folium. Ask the user whether these maps
  exist and where.
- The midline: about section 75 (D99 x = 0 in J20/P20). Sections 0-75 are about 3.75 mm of the other side.

## 6. Proposed work plan (after the decision)

1. **Ground truth first.** Ask the user (or Mat) to mark the lobules on about 4 sections by hand: one at the
   midline, one in the paravermis, two in the hemisphere. Also mark the vermis edge on a few sections. Without
   this, no method can be measured.
2. **Settle the lateral coverage** (§3 item 3). If the section-axis scale changes, refit the reference affine
   (`mri_register.py --no-reference ... --save-reference`, check by eye) and run the labels again.
3. **Fissure map.** From the stack density at rescale 20: cortex mask, molecular-layer / fissure map, and a 3D
   skeleton of the white matter. Check it on the midline section, where all ten vermal lobules are visible.
4. **Build the chosen option** as a new script beside `atlas_labels.py`, with the same output contract (uint8 on
   the stack grid, a names table, a QC section). Do not change the meaning of the existing files; add new ones.
5. **Validate** against the hand labels: Dice per lobule, boundary distance (mm), and continuity of each label
   across sections. Compare with the current D99 propagation as the baseline.
6. **Run at rescale 5** on the HPC result (`run_..._hpc.py` output). Document memory and time in §7.3.

## 7. Constraints

- Do not change the original scripts or existing output formats; Mat's processing depends on them (§3). Add new
  files and document every change in `REGISTRATION_NOTES_local.md`.
- Use uv (`uv run python ...`). On the workstation, other jobs have priority: nice 19, few workers, our processes
  are killed first under memory pressure (`env_compat.py`).
- The D99 files are not in git (`local_extras/`). Copy them to any new machine.
- Commit on the branch with clear messages. Write user-facing text in ASD-STE100 Simplified Technical English.
