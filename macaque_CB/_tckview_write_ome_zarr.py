"""Helper for export_ome_zarr.py: runs in tck_view's uv environment (neuralarc) and writes the OME-Zarr.

usage: uv run --project ~/Documents/code/tck_view python _tckview_write_ome_zarr.py <spec.json>
"""
import json
import sys

import numpy as np
from neuralarc.core import ome_zarr


def main(spec_f):
    with open(spec_f) as f:
        spec = json.load(f)
    affine = np.asarray(spec['affine'], float)
    density = np.load(spec['density'], mmap_mode='r')
    image = ome_zarr.Volume(density, affine, name=spec['name'], space_unit='millimeter',
                            source=spec['source'], anatomical=spec['anatomical'], reduction='mean')
    labels = []
    for lab in spec['labels']:   # [{name, path}], uint8 arrays on the image grid; empty ones are left out
        arr = np.load(lab['path'], mmap_mode='r')
        if arr.any():
            labels.append(ome_zarr.Volume(arr, affine, name=lab['name'], space_unit='millimeter',
                                          source=spec['source'], anatomical=spec['anatomical']))
    summary = ome_zarr.write_ome_zarr(spec['out'], image, kind='image', labels=labels, overwrite=spec['overwrite'],
                                      version='0.6', attrs=spec['attrs'], progress=print)
    print(json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool, list))}, default=str))

    # read back the full-resolution level and check it against what was written
    data, back_affine, _ = ome_zarr.read_ome_zarr(spec['out'], level=0)
    # an axis-aligned volume is stored reoriented to RAS+, so compare through world coordinates
    same_values = np.isclose(np.sort(np.asarray(data, np.float64).ravel())[::997],
                             np.sort(np.asarray(density, np.float64).ravel())[::997], rtol=1e-6).all()
    corners = np.array([[0, 0, 0, 1], [density.shape[0] - 1, density.shape[1] - 1, density.shape[2] - 1, 1]], float)
    world = (affine @ corners.T).T[:, :3]
    corners_back = np.array([[0, 0, 0, 1], [data.shape[0] - 1, data.shape[1] - 1, data.shape[2] - 1, 1]], float)
    world_back = np.sort((back_affine @ corners_back.T).T[:, :3], axis=0)
    print(f'read back: shape {data.shape} (written {density.shape}), values match: {same_values}, '
          f'world extent match: {np.allclose(np.sort(world, axis=0), world_back, atol=1e-6)}')


if __name__ == '__main__':
    main(sys.argv[1])
