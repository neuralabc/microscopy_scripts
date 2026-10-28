"""Export a registered density stack as OME-Zarr (NGFF 0.6, Zarr v3) in the format that tck_view (neuralarc) reads.

The pipeline stacks hold cell counts per voxel (downsample_image sums blocks of the 10 um count maps).
They are written as densities in cells/mm^3 (count / voxel volume), so that the averaged pyramid levels
stay densities and runs at different rescales are comparable.

Placement:
  stack  the registered stack grid in mm (in-plane rescale*10 um, 50 um sections), no anatomical claim
  mri    the same voxels (not resampled) with an NGFF 0.6 affine to D99 world coordinates (RAS, mm),
         from --mri-affine (made by mri_register.py, which runs this export itself) or else from a fresh
         shear-free affine fit of this stack to the D99 cerebellum template (mri_anchor.py)

Slices that were interpolated (missing_idxs_to_fill) are marked in a label map `synthetic_slices` (1 = interpolated)
under the image's labels/ group: their densities are not measured.

The OME-Zarr itself is written by neuralarc.core.ome_zarr.write_ome_zarr (default chunks and pyramid: levels
halve the finest axes first, then all, until the largest side is at most 256), run in tck_view's own uv
environment (it needs Python >= 3.13 and its own dependencies) through _tckview_write_ome_zarr.py.

usage (from the repo's uv environment):
  uv run python macaque_CB/export_ome_zarr.py --run-dir <output_dir> --rescale 5 --placement mri --out <path>.ome.zarr
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import nibabel

SUBJECT = 'zefir'
DEFAULT_LEVEL = 'coreg12nl_win12_rigsyn_4_groupwise_iter9'
DEFAULT_MISSING = [32, 59, 120, 160, 189, 228]
SLICE_MM = 0.05
TCK_VIEW = os.path.expanduser('~/Documents/code/tck_view')
HERE = os.path.dirname(os.path.abspath(__file__))


def stack_affine_mm(shape, res_mm, slice_mm):
    """Voxel -> mm, centred like create_affine (same grid as the pipeline stack, in mm instead of um)."""
    aff = np.diag([res_mm, res_mm, slice_mm, 1.0])
    aff[:3, 3] = -(np.array(shape[:3]) - 1) / 2.0 * np.array([res_mm, res_mm, slice_mm])
    return aff


def mri_world_affine(stack, res_mm, slice_mm, fit_dir, direction=1, affine_file=None):
    """Voxel -> D99 world (mm), from a shear-free affine fit of the stack to the MRI, or from a given affine
    (ANTs .mat, MRI frame -> stack frame, e.g. from mri_register.py) without a fit."""
    import env_compat  # noqa: F401
    import ants
    import mri_anchor
    from evaluate_registration import MRI_T, MRI_M
    if affine_file is None:
        aff_f, qc = mri_anchor.stack_to_mri_affine(stack, res_mm, slice_mm, MRI_T, MRI_M, fit_dir, direction=direction)
    else:
        aff_f = affine_file
        p = np.array(ants.read_transform(aff_f).parameters)
        qc = dict(direction=direction, x0_mm=None, affine_file=os.path.abspath(aff_f),
                  scales_stack_axes=[float(v) for v in np.linalg.svd(p[:9].reshape(3, 3), compute_uv=False)],
                  dice_affine=None)
    tx = ants.read_transform(aff_f)  # MRI frame (index * spacing) -> stack frame (voxel * spacing)
    pts_m = np.array([[x, y, z] for x in (0, 50) for y in (0, 50) for z in (0, 50)], float)
    pts_s = np.array([tx.apply_to_point(p) for p in pts_m])
    A = np.linalg.lstsq(np.c_[pts_s, np.ones(len(pts_s))], pts_m, rcond=None)[0].T  # stack mm -> MRI frame mm
    stack_to_mri = np.vstack([A, [0, 0, 0, 1]])
    mri = nibabel.load(MRI_T)
    sp = float(mri.header.get_zooms()[0])
    vox_to_stackmm = np.diag([res_mm, res_mm, slice_mm, 1.0])
    mriframe_to_world = mri.affine @ np.diag([1 / sp, 1 / sp, 1 / sp, 1.0])
    return mriframe_to_world @ stack_to_mri @ vox_to_stackmm, qc


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--level', default=DEFAULT_LEVEL, help='registration level of the stack to export')
    p.add_argument('--placement', choices=('stack', 'mri'), default='stack')
    p.add_argument('--out', required=True, help='output .ome.zarr path')
    p.add_argument('--missing', default=','.join(map(str, DEFAULT_MISSING)),
                   help='interpolated slice indices (comma separated, empty for none)')
    p.add_argument('--direction', type=int, choices=(1, -1), default=1,
                   help='mri placement: hemisphere of D99 the stack is placed in (+1: right, as the pipeline default)')
    p.add_argument('--mri-affine', default=None,
                   help='mri placement: use this affine (ANTs .mat from mri_register.py) instead of a new fit; '
                        'a new fit can differ between runs (one of four was turned by 180 deg, 2026-10-28)')
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--tck-view', default=TCK_VIEW)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()

    run_dir = a.run_dir.rstrip('/') + '/'
    stack_f = f'{run_dir}{a.subject}_{a.level}_stack.nii.gz'
    counts = nibabel.load(stack_f).get_fdata(dtype=np.float32)
    res_mm = a.rescale * 10 / 1000.0
    voxel_mm3 = res_mm * res_mm * SLICE_MM
    density = counts / np.float32(voxel_mm3)

    missing = [int(v) for v in a.missing.split(',') if v.strip()]
    synthetic = np.zeros(counts.shape, np.uint8)
    for s in missing:
        if s < counts.shape[2]:
            synthetic[..., s] = 1

    out = os.path.abspath(a.out)
    attrs = {'microscopy_scripts': {
        'quantity': 'cell density', 'unit': 'cells/mm^3',
        'stored_as': 'pipeline counts per voxel / voxel volume',
        'voxel_volume_mm3': voxel_mm3, 'counts_per_voxel_to_density': 1.0 / voxel_mm3,
        'run_dir': run_dir, 'level': a.level, 'stack_file': os.path.basename(stack_f),
        'rescale': a.rescale, 'in_plane_mm': res_mm, 'section_mm': SLICE_MM,
        'placement': a.placement, 'synthetic_slices': missing}}
    if a.placement == 'mri':
        affine, qc = mri_world_affine(counts, res_mm, SLICE_MM, out + '.mri_fit', direction=a.direction,
                                     affine_file=a.mri_affine)
        # The stack axes are (in-plane, in-plane, section) and the sections are sagittal, so array axis 2 runs
        # left-right in the world, and the fit includes an in-plane flip (determinant < 0). tck_view's slice panes
        # take array axes 0, 1, 2 as world x, y, z, and the axial pane showed the wrong plane. So the array is
        # permuted and flipped (no resampling) to the closest RAS+ order; only a small rotation stays in the affine.
        ornt = nibabel.orientations.io_orientation(affine)
        reorient = nibabel.orientations.inv_ornt_aff(ornt, density.shape)
        density = nibabel.orientations.apply_orientation(density, ornt)
        synthetic = nibabel.orientations.apply_orientation(synthetic, ornt)
        affine = affine @ reorient
        if np.linalg.det(affine[:3, :3]) <= 0 or not np.all(np.argmax(np.abs(affine[:3, :3]), axis=0) == [0, 1, 2]):
            raise ValueError(f'reorientation to RAS+ failed:\n{affine}')
        attrs['microscopy_scripts'].update(
            world='D99 cerebellum_brainstem_template_crop (RAS, mm)',
            mri_fit={k: qc.get(k) for k in ('direction', 'x0_mm', 'scales_stack_axes', 'dice_affine', 'affine_file')},
            # row k: stack axis k (i, j, section) is array axis ornt[k][0], reversed if ornt[k][1] == -1
            array_axes_from_stack=ornt.tolist())
        anatomical = True
    else:
        affine = stack_affine_mm(counts.shape, res_mm, SLICE_MM)
        anatomical = False

    with tempfile.TemporaryDirectory(prefix='ome_zarr_export_', dir=os.path.dirname(out)) as tmp:
        np.save(os.path.join(tmp, 'density.npy'), density)
        np.save(os.path.join(tmp, 'synthetic.npy'), synthetic)
        spec = dict(out=out, density=os.path.join(tmp, 'density.npy'), synthetic=os.path.join(tmp, 'synthetic.npy'),
                    affine=affine.tolist(), anatomical=anatomical, name=f'{a.subject}_cell_density',
                    source=stack_f, attrs=attrs, overwrite=a.overwrite)
        spec_f = os.path.join(tmp, 'spec.json')
        with open(spec_f, 'w') as f:
            json.dump(spec, f)
        env = {k: v for k, v in os.environ.items() if k not in ('VIRTUAL_ENV', 'PYTHONPATH', 'UV_PROJECT_ENVIRONMENT')}
        subprocess.run(['uv', 'run', '--project', a.tck_view, 'python',
                        os.path.join(HERE, '_tckview_write_ome_zarr.py'), spec_f], check=True, env=env)


if __name__ == '__main__':
    main()
