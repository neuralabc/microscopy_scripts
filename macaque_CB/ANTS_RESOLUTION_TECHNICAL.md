# Resolution Support in ANTs Registration Functions

## Summary

This document provides detailed technical information about how resolution/spacing information is handled by the ANTs and nighres registration functions used in this codebase.

## Background on ANTs Resolution Handling

### ANTs Image Objects

ANTs (Advanced Normalization Tools) stores spacing/resolution information in image objects via:
- **Spacing**: Physical size of voxels (e.g., 0.4×0.4×0.05 mm)
- **Origin**: Physical location of the first voxel in world coordinates
- **Direction**: Orientation matrix defining image axes

When you read a NIfTI file with `ants.image_read()`, ANTs automatically loads this information from the file's header and affine matrix.

### ANTs Registration Behavior

ANTs registration functions **automatically use** the spacing information from the image objects during registration. This means:

1. **Metric calculations** are performed in physical space
2. **Transform parameters** are defined in physical coordinates
3. **Regularization** is applied based on physical distances

There is **no `ignore_res` parameter** in ANTs' native Python API (ANTsPy). The spacing is always considered.

## Implementation Strategy

Since ANTs doesn't have an `ignore_res` flag like nighres, we implement resolution control differently:

### For Nighres (`do_reg` function)

```python
def do_reg(..., use_resolution=False):
    reg = nighres.registration.embedded_antspy_2d_multi(
        ...
        ignore_res=not use_resolution,  # Direct control via nighres flag
        ...
    )
```

Nighres provides an `ignore_res` parameter that tells the underlying ANTs call to ignore spacing information.

### For ANTs Direct (`do_reg_ants` function)

```python
def do_reg_ants(..., use_resolution=False):
    source_img = ants.image_read(source)
    target_img = ants.image_read(target)
    
    # If not using resolution, reset spacing to (1,1,1)
    if not use_resolution:
        source_img.set_spacing((1.0, 1.0, 1.0))
        target_img.set_spacing((1.0, 1.0, 1.0))
    
    # Now ANTs will use the spacing we set (either original or 1×1×1)
    rigid_reg = ants.registration(
        fixed=target_img,
        moving=source_img,
        type_of_transform='Rigid',
        ...
    )
```

By explicitly resetting the spacing to (1, 1, 1) before registration, we effectively make ANTs treat the images as having unit voxels, which is equivalent to ignoring the original resolution.

## Verification of Compatibility

### What We Checked

1. ✅ **ANTs `image_read()` preserves spacing**: Confirmed by reading NIfTI files and checking `img.spacing`
2. ✅ **`set_spacing()` modifies ANTs images**: Can change spacing before registration
3. ✅ **Registration uses spacing**: ANTs documentation and testing confirm this
4. ✅ **Both nighres and ANTs paths work**: The implementation handles both registration backends

### Why This Approach Works

**With `use_resolution=False` (default):**
```
1. NIfTI created with 1×1×1 spacing in affine matrix
2. ANTs reads file → spacing is (1, 1, 1) OR spacing is reset to (1, 1, 1)
3. Registration operates in normalized voxel space
4. Transforms are in voxel coordinates
5. Result: Same behavior as original code
```

**With `use_resolution=True`:**
```
1. NIfTI created with actual resolution (e.g., 0.4×0.4×0.05) in affine matrix
2. ANTs reads file → spacing is (0.4, 0.4, 0.05)
3. Registration operates in physical space (mm)
4. Transforms are in physical coordinates
5. Result: Resolution-aware registration
```

## Testing Recommendations

To verify that ANTs properly handles resolution in your specific environment:

### Test 1: Check Image Spacing

```python
import ants

# Create test image with known spacing
img = ants.image_read('test_slice.nii.gz')
print(f"Original spacing: {img.spacing}")

# Reset spacing
img.set_spacing((1.0, 1.0, 1.0))
print(f"Reset spacing: {img.spacing}")

# Spacing should now be (1, 1, 1)
assert img.spacing == (1.0, 1.0, 1.0)
```

### Test 2: Compare Registrations

```python
import ants

# Load two test slices
source = ants.image_read('slice_001.nii.gz')
target = ants.image_read('slice_002.nii.gz')

# Test with original spacing
reg1 = ants.registration(fixed=target, moving=source, type_of_transform='Rigid')

# Test with reset spacing
source.set_spacing((1.0, 1.0, 1.0))
target.set_spacing((1.0, 1.0, 1.0))
reg2 = ants.registration(fixed=target, moving=source, type_of_transform='Rigid')

# The transformations should differ slightly due to spacing
print(f"Transform 1: {reg1['fwdtransforms']}")
print(f"Transform 2: {reg2['fwdtransforms']}")
```

## Known Limitations

### 1. Nighres Wrapper Behavior

The nighres wrapper (`embedded_antspy_2d_multi`) may have its own quirks in how it handles the `ignore_res` parameter. Our implementation relies on this parameter working as documented.

### 2. Transform File Compatibility

When switching between resolution modes, transform files may not be directly compatible:
- Transforms from `use_resolution=False` are in voxel coordinates
- Transforms from `use_resolution=True` are in physical (mm) coordinates

Do not mix transforms from different modes.

### 3. Deformation Field Handling

The code that creates intermediate slices using deformation fields (lines ~672-745 in `slice_registration_functions.py`) manually preserves spacing:

```python
avg_field = ants.from_numpy(..., spacing=pre_to_post_field.spacing+(1.0,))
new_image.set_spacing(pre_ants.spacing)
```

This code should work correctly with both resolution modes, but has not been extensively tested with `use_resolution=True`.

## References

- **ANTsPy Documentation**: https://antspy.readthedocs.io/
- **ANTs GitHub**: https://github.com/ANTsX/ANTs
- **Nighres Documentation**: https://nighres.readthedocs.io/
- **NIfTI Format**: https://nifti.nimh.nih.gov/

## Questions?

If you encounter issues with resolution handling:

1. Verify your ANTsPy version: `python -c "import ants; print(ants.__version__)"`
2. Check that NIfTI files have correct affine matrices: `nibabel.load('file.nii.gz').affine`
3. Test with a small subset of data before full pipeline runs
4. Compare visual results between `use_resolution=True` and `use_resolution=False`

The default (`use_resolution=False`) has been empirically validated to produce better results for this specific 2D slice registration pipeline.
