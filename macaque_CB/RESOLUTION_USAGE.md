# Resolution Information in Registration

## Overview

The registration pipeline now supports optional inclusion of resolution/spacing information during the registration process. By default, images are treated as having 1×1×1 voxel spacing during registration (the original behavior), but you can optionally enable the use of actual resolution information.

## Configuration

In `run_slice_registration.py`, set the following parameter near the top of the file (around line 40):

```python
# Control whether to use resolution information during registration
# When False (default), images are treated as 1x1x1 during registration (better empirical results)
# When True, resolution information is included in the registration process
use_resolution_in_registration = False  # Change to True to enable resolution
```

## How It Works

### With `use_resolution_in_registration = False` (Default)

1. **NIfTI Creation**: Images are created with an identity affine matrix (1×1×1 voxel spacing)
2. **Registration**: 
   - For nighres-based registration (`do_reg`): Uses `ignore_res=True` flag
   - For ANTs-based registration (`do_reg_ants`): Resets spacing to (1.0, 1.0, 1.0)
3. **Result**: Registration operates in normalized voxel space

### With `use_resolution_in_registration = True`

1. **NIfTI Creation**: Images are created with resolution information in the affine matrix:
   - `affine[0,0] = in_plane_res_x` (typically 0.4 mm after rescaling)
   - `affine[1,1] = in_plane_res_y` (typically 0.4 mm after rescaling)
   - `affine[2,2] = in_plane_res_z` (typically 0.05 mm after rescaling)
   - Also sets zooms via `nifti.set_zooms()`
2. **Registration**:
   - For nighres-based registration: Uses `ignore_res=False` flag
   - For ANTs-based registration: Preserves original spacing from image headers
3. **Result**: Registration operates in physical space with actual resolution

## When to Use Resolution Information

### Use Resolution (`True`) When:

- ✅ Working with anisotropic data (different X/Y/Z resolutions)
- ✅ Comparing registrations across datasets with different resolutions
- ✅ Need physical measurements in real-world units (mm, microns)
- ✅ Integrating with other tools that expect proper resolution information
- ✅ Resolution varies significantly across the dataset

### Don't Use Resolution (`False`) When:

- ✅ **Default recommendation** - Empirical testing shows better registration results
- ✅ All slices have uniform, isotropic in-plane resolution (like 10×10 microns)
- ✅ Following the tested pipeline that has been validated
- ✅ Resolution differences are minimal and won't affect registration quality

## Technical Details

### Affected Functions

The following functions now accept a `use_resolution` parameter:

1. **`do_reg()`** - Controls `ignore_res` parameter for nighres
2. **`do_reg_ants()`** - Controls whether spacing is reset to (1,1,1)
3. **`coreg_single_slice_orig()`** - Passes parameter through to registration
4. **`run_parallel_coregistrations()`** - Passes parameter to all parallel registrations
5. **`run_cascading_coregistrations_v2()`** - Passes parameter to cascading registrations
6. **`do_initial_translation_reg()`** - Passes parameter to initial translation step

### Why Default is `False`

From the original codebase comment (line 38 in `run_slice_registration.py`):
> "registration itself performs much better when we do not specify the res"

This was determined through empirical testing. The registration algorithms appear to work better when operating in normalized voxel space rather than physical space, especially for the 2D slice-to-slice registration approach used in this pipeline.

## Testing Your Choice

To determine which setting works best for your data:

1. Run a subset of your data with `use_resolution_in_registration = False`
2. Run the same subset with `use_resolution_in_registration = True`
3. Compare:
   - Visual alignment quality
   - Registration convergence
   - Mutual information scores
   - Overall stack coherence

## Example

```python
# In run_slice_registration.py

# Original resolution (before rescaling)
in_plane_res_x = 10  # 10 microns/pixel
in_plane_res_y = 10  # 10 microns/pixel
in_plane_res_z = 50  # 50 microns (slice thickness)

# After rescaling by factor of 40
rescale = 40
in_plane_res_x = rescale * in_plane_res_x / 1000  # 0.4 mm
in_plane_res_y = rescale * in_plane_res_y / 1000  # 0.4 mm
in_plane_res_z = in_plane_res_z / 1000           # 0.05 mm

# Enable resolution in registration
use_resolution_in_registration = True  # Use actual 0.4×0.4×0.05 mm spacing

# The rest of the pipeline automatically uses this setting
```

## Additional Notes

- The resolution setting only affects the registration process itself
- Template generation and output always use the specified `voxel_res` for proper physical spacing
- Changing this setting does not require any other code modifications
- All registration functions maintain backward compatibility with existing code
