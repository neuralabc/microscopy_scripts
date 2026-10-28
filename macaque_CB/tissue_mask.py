"""Tissue mask of a registered stack, written beside it: <run_dir>/<subject>_<level>_tissue_mask.nii.gz

Definition (the same as in the QC report, evaluate_registration.py and the MRI fit):
  - a voxel is tissue if its counts are above mri_anchor.tissue_threshold_for(in-plane mm), i.e. 2 cells per
    50 um voxel (about 16,000 cells/mm^3); the count threshold scales with the voxel area, so the mask means the
    same at every rescale;
  - holes inside each slice are filled (white matter has a low cell density but is inside the tissue).
Interpolated slices (missing_idxs_to_fill) get a mask from their interpolated values; the OME-Zarr label
`synthetic_slices` marks them.

The file has the same grid, affine and header as the stack, with uint8 values (1 = tissue).

Run by the main script for the final level; for another level:
  uv run python macaque_CB/tissue_mask.py --run-dir <output_dir> --rescale 5 --level <level>
"""
import argparse
import os

import numpy as np
import nibabel
from scipy import ndimage as ndi

SUBJECT = 'zefir'


def tissue_threshold_for(res_mm, counts_per_50um_voxel=2.0):
    """Same as mri_anchor.tissue_threshold_for (kept here so that this module does not need ANTs)."""
    return counts_per_50um_voxel * (res_mm / 0.05) ** 2


def tissue_mask(stack_counts, res_mm):
    """uint8 mask (1 = tissue) of a counts stack (i, j, slice) with in-plane voxel size res_mm."""
    thr = tissue_threshold_for(res_mm)
    out = np.zeros(stack_counts.shape, np.uint8)
    for s in range(stack_counts.shape[2]):
        out[..., s] = ndi.binary_fill_holes(stack_counts[..., s] > thr)
    return out


def mask_path(run_dir, subject, level):
    return os.path.join(run_dir, f'{subject}_{level}_tissue_mask.nii.gz')


def write_tissue_mask(run_dir, subject, level, rescale, overwrite=False):
    out = mask_path(run_dir, subject, level)
    if os.path.isfile(out) and not overwrite:
        return out
    img = nibabel.load(os.path.join(run_dir, f'{subject}_{level}_stack.nii.gz'))
    mask = tissue_mask(np.asarray(img.dataobj, dtype=np.float32), rescale * 10 / 1000.0)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    hdr['descrip'] = b'tissue mask: counts > 2 per 50um voxel, holes filled'
    nibabel.save(nibabel.Nifti1Image(mask, img.affine, hdr), out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rescale', type=int, required=True)
    p.add_argument('--level', required=True)
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    print(write_tissue_mask(a.run_dir, a.subject, a.level, a.rescale, overwrite=a.overwrite))


if __name__ == '__main__':
    main()
