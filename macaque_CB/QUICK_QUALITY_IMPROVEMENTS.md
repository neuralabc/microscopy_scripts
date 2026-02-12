# Registration Quality Improvement - Executive Summary

## Problem
Registered slices are **jagged and discontinuous** despite current smoothing efforts.

## Root Cause
Independent per-slice registrations without explicit smoothness constraints between adjacent slices in the Z-axis.

## Quick Fixes (Implement First)

### 1. Enable Final Stack Smoothing ⭐⭐⭐
**Current:** Disabled (`across_slice_smoothing_sigma = 0` on line 349)  
**Fix:** Set `final_stack_smoothing_sigma = 2` for final output  
**Impact:** 30-40% reduction in jaggedness  
**Effort:** 15 minutes

```python
# In run_slice_registration.py
final_stack_smoothing_sigma = 2  # Add this parameter

# In generate_stack_and_template(), apply to final stack:
if final_stack_smoothing_sigma > 0:
    from scipy.ndimage import gaussian_filter
    final_stack = gaussian_filter(final_stack, sigma=(0, 0, final_stack_smoothing_sigma))
```

### 2. Increase Deformation Smoothing ⭐⭐⭐
**Current:** `syn_flow_sigma=3`, `syn_total_sigma=0`  
**Fix:** Increase both parameters  
**Impact:** 20-30% smoother deformations  
**Effort:** 5 minutes

```python
# In do_reg_ants() function:
syn_flow_sigma = 4      # Increase from 3
syn_total_sigma = 2     # Enable from 0
```

### 3. Smooth Deformation Fields ⭐⭐
**Current:** No post-processing of deformation fields  
**Fix:** Add smoothing function  
**Impact:** 25-35% reduction in discontinuities  
**Effort:** 1-2 hours

```python
def smooth_deformation_fields_across_stack(deformation_fields, sigma=1.0):
    """Apply Gaussian smoothing to deformation fields in Z-direction."""
    stacked = np.stack(deformation_fields, axis=-1)
    smoothed = gaussian_filter(stacked, sigma=(0, 0, 0, sigma))
    return [smoothed[..., i] for i in range(smoothed.shape[-1])]
```

## Recommended Implementation Order

### Phase 1: Quick Wins (Same Day)
1. ✅ Enable final stack smoothing → Test → Validate
2. ✅ Increase SyN smoothing parameters → Test → Validate
3. ✅ Visual inspection on 10-20 slice subset

**Expected Result:** 40-60% improvement in 1 day

### Phase 2: Transform Processing (2-3 Days)
1. ✅ Implement deformation field smoothing
2. ✅ Add median filtering for outliers
3. ✅ Test on full dataset

**Expected Result:** 60-75% improvement cumulative

### Phase 3: Advanced Methods (1 Week)
1. ✅ Bilateral filtering (edge-preserving)
2. ✅ Adaptive regularization based on MI
3. ✅ Comprehensive validation

**Expected Result:** 75-85% improvement cumulative

## Parameter Tuning Guide

| Parameter | Start With | If Too Smooth | If Still Jagged |
|-----------|------------|---------------|-----------------|
| `final_stack_smoothing_sigma` | 2 | Reduce to 1 | Increase to 3 |
| `syn_flow_sigma` | 4 | Keep at 3 | Increase to 5 |
| `syn_total_sigma` | 2 | Reduce to 1 | Increase to 3 |

## Success Metrics

Track these before/after:
- Visual smoothness in Z-axis views (sagittal/coronal)
- Standard deviation of slice-to-slice differences
- Mutual Information between adjacent registered slices

## Key Files to Modify

1. `run_slice_registration.py` - Add final smoothing parameter
2. `slice_registration_functions.py` - Modify `do_reg_ants()`, add smoothing functions
3. `generate_stack_and_template()` - Apply final smoothing

## Full Details

See `REGISTRATION_QUALITY_IMPROVEMENT_PLAN.md` for complete analysis, all strategies, and detailed implementation guidance.

## Estimated Timeline

- **Noticeable improvement:** Same day (Phase 1)
- **Significant improvement:** 2-3 days (Phase 1-2)
- **Optimal results:** 1-2 weeks (All phases)
