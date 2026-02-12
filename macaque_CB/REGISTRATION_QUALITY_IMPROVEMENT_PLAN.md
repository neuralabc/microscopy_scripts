# Registration Quality Improvement Plan

## Problem Statement

After registration, slices remain **fairly jagged and discontinuous**, indicating that independent per-slice registrations are not sufficiently constrained to produce smooth, coherent 3D volumes.

## Root Cause Analysis

### Primary Issues

1. **Independent Per-Slice Registration**
   - Each slice registers independently to its neighbors
   - No explicit smoothness constraint between adjacent slices' deformation fields
   - Results in discontinuous transformations across the Z-axis

2. **Smoothing Disabled on Final Output**
   - Line 349: `across_slice_smoothing_sigma = 0` for final stack
   - Smoothing only applied during intermediate template creation
   - Final registered slices have full jaggedness visible

3. **Known Problematic Approach**
   - Line 17 comment: "nonlinear slice templates... result in very jagged registrations"
   - Nonlinear interpolation between slices can overfit
   - Currently disabled but identified as a quality issue

4. **Limited Deformation Field Smoothing**
   - Current: `syn_flow_sigma = 3`, `syn_total_sigma = 0`
   - No post-registration smoothing of deformation fields
   - No spatial regularization across the stack

## Current Quality Control Mechanisms

### What's Already in Place

✅ **Across-Slice Smoothing** (during template creation)
- Parameter: `across_slice_smoothing_sigma = 5`
- Applied: Gaussian filter in Z-direction only
- Timing: After stacking, before template creation
- **Issue:** Disabled on final output (line 349)

✅ **Regularization Control**
- Medium regularization for initial registrations
- High regularization for repeated SyN runs (line 580)
- Controls deformation smoothness within slices

✅ **Cascading Registration**
- Multiple iterations with different anchor points
- Progressive window expansion: 3 → 6 → 9 slices
- Weighted forward/reverse registration

✅ **Iteration Limits**
- Rigid: 5000 iterations
- Coarse: 2000, Medium: 1000, Fine: 200
- Convergence: 1e-6
- Note: Fine iterations reduced from 500 (too aggressive)

✅ **Per-Slice Templates**
- Each slice uses median of itself + adjacent slices
- Provides local anchoring
- **Issue:** May cause slice-to-slice discontinuities

## Improvement Strategies

### Priority 1: High Impact, Low Effort

#### 1.1 Enable Final Stack Smoothing ⭐⭐⭐
**Impact:** High | **Effort:** Low | **Risk:** Low

**Current State:**
```python
across_slice_smoothing_sigma = 0  # Line 349 - disabled for final output
```

**Recommendation:**
```python
final_stack_smoothing_sigma = 2  # Gentle smoothing for continuity
```

**Implementation:**
- Add parameter for final output smoothing
- Apply Gaussian smoothing in Z-direction only: `sigma=(0, 0, final_sigma)`
- Keep X-Y resolution intact, smooth only Z-discontinuities
- Start with sigma=2, tune based on results

**Benefits:**
- Direct reduction in Z-axis jaggedness
- Maintains in-plane detail
- Computationally cheap
- Easy to revert/tune

---

#### 1.2 Increase Deformation Field Smoothing ⭐⭐⭐
**Impact:** High | **Effort:** Low | **Risk:** Medium

**Current State:**
```python
syn_flow_sigma = 3       # Flow field smoothing
syn_total_sigma = 0      # Total field smoothing (disabled)
```

**Recommendation:**
```python
syn_flow_sigma = 4       # Increase from 3 → 4
syn_total_sigma = 2      # Enable total field smoothing
```

**Implementation:**
- Modify `do_reg_ants()` function parameters
- Test values: flow_sigma ∈ [3, 4, 5], total_sigma ∈ [0, 1, 2]
- Monitor impact on alignment quality vs smoothness

**Benefits:**
- Smoother deformations within each slice
- Reduces local overfitting
- Built-in ANTs parameter

**Risks:**
- May reduce alignment precision
- Need to validate MI scores don't degrade

---

#### 1.3 Add Post-Registration Deformation Field Smoothing ⭐⭐
**Impact:** Medium-High | **Effort:** Medium | **Risk:** Low

**Current State:**
- No post-processing of deformation fields
- TODO comment at line 206: "add deformation_smoothing across the stack"

**Recommendation:**
```python
def smooth_deformation_fields_across_stack(deformation_fields, sigma=1.0):
    """
    Apply Gaussian smoothing to deformation fields in Z-direction.
    
    Parameters:
        deformation_fields: List of 2D/3D deformation fields
        sigma: Smoothing sigma for Z-direction
    
    Returns:
        Smoothed deformation fields
    """
    # Stack deformation fields
    stacked = np.stack(deformation_fields, axis=-1)
    
    # Smooth only in Z-direction (last axis)
    smoothed = gaussian_filter(stacked, sigma=(0, 0, 0, sigma))
    
    # Unstack back to list
    return [smoothed[..., i] for i in range(smoothed.shape[-1])]
```

**Implementation:**
- Apply after all registrations complete
- Before final template generation
- Smooth forward and inverse transform fields
- Re-apply smoothed transforms to images

**Benefits:**
- Enforces smoothness across slices
- Reduces registration "jitter"
- Preserves registration quality

---

### Priority 2: Medium Impact, Medium Effort

#### 2.1 Adaptive Regularization Based on MI ⭐⭐
**Impact:** Medium | **Effort:** Medium | **Risk:** Low

**Current State:**
- Fixed regularization: Medium → High
- TODO at line 32: "potentially scale regularization"

**Recommendation:**
```python
def compute_adaptive_regularization(mi_score, mi_threshold=0.3):
    """
    Adjust regularization based on registration confidence.
    
    Low MI (poor alignment) → High regularization (more smoothing)
    High MI (good alignment) → Medium regularization (preserve detail)
    """
    if mi_score < mi_threshold:
        return 'High'   # Problem slice, constrain heavily
    elif mi_score < 0.5:
        return 'Medium'
    else:
        return 'Low'    # Good alignment, allow detail
```

**Implementation:**
- Compute MI after initial registration
- Adjust regularization for subsequent iterations
- Log which slices receive high regularization
- Useful for identifying problem regions

**Benefits:**
- Targeted smoothing where needed
- Preserves detail in well-aligned regions
- Diagnostic tool for quality assessment

---

#### 2.2 Bilateral Filtering in Z-Direction ⭐⭐
**Impact:** Medium | **Effort:** Medium | **Risk:** Low

**Current State:**
- Only Gaussian smoothing used
- No edge-preserving filters

**Recommendation:**
```python
from scipy.ndimage import generic_filter

def bilateral_filter_z(volume, sigma_spatial=2, sigma_intensity=0.1):
    """
    Apply bilateral filter in Z-direction only.
    Preserves sharp boundaries while smoothing gradual changes.
    """
    # Implement or use existing bilateral filter
    # Apply only along Z-axis to preserve in-plane features
```

**Implementation:**
- Apply to final registered stack
- Preserves tissue boundaries while smoothing noise
- More sophisticated than Gaussian smoothing

**Benefits:**
- Edge-preserving smoothing
- Better than pure Gaussian for biological structures
- Reduces jaggedness without excessive blur

---

#### 2.3 Median Filtering of Deformation Fields ⭐⭐
**Impact:** Medium | **Effort:** Low | **Risk:** Low

**Current State:**
- No outlier removal in deformation fields

**Recommendation:**
```python
from scipy.ndimage import median_filter

def remove_deformation_outliers(deformation_field, kernel_size=3):
    """
    Apply median filter to remove outlier deformations.
    """
    return median_filter(deformation_field, size=(kernel_size, kernel_size, 1))
```

**Implementation:**
- Apply to deformation fields before final application
- Removes registration "spikes"
- Small kernel (3x3) to preserve structure

**Benefits:**
- Removes outlier registrations
- Reduces local artifacts
- Computationally cheap

---

### Priority 3: High Impact, High Effort

#### 3.1 Implement Sliding Window Registration ⭐⭐⭐
**Impact:** High | **Effort:** High | **Risk:** Medium

**Current State:**
- Cascading registration with fixed windows
- Windows expand but don't slide smoothly

**Recommendation:**
```python
def sliding_window_registration(slices, window_size=5, stride=1):
    """
    Register slices using overlapping sliding windows.
    
    Window 1: slices [0-4]
    Window 2: slices [1-5] (overlap with window 1)
    Window 3: slices [2-6] (overlap with window 2)
    ...
    
    Average transformations in overlap regions.
    """
```

**Implementation:**
- Register groups of slices with overlap
- Average deformations in overlapping regions
- Provides natural smoothness constraint
- More complex bookkeeping

**Benefits:**
- Inherent smoothness across boundaries
- Better global consistency
- Reduces independent slice artifacts

**Risks:**
- Significantly more complex
- May require refactoring registration pipeline
- Increased computation time

---

#### 3.2 Global 3D Regularization Term ⭐⭐⭐
**Impact:** Very High | **Effort:** Very High | **Risk:** High

**Current State:**
- 2D slice-by-slice registration
- No 3D spatial regularization

**Recommendation:**
```python
def add_3d_smoothness_constraint(transforms, lambda_smooth=0.1):
    """
    Minimize: E_data + lambda_smooth * E_smoothness
    
    E_smoothness = sum over adjacent slices:
        ||transform[i] - transform[i+1]||^2
    """
```

**Implementation:**
- Requires optimization framework
- Post-processing step that adjusts transforms
- Minimize data term + smoothness term
- Could use variational methods or optimization

**Benefits:**
- Theoretically optimal approach
- Enforces true 3D smoothness
- Reduces jaggedness at fundamental level

**Risks:**
- Very complex implementation
- May require custom optimization code
- Computationally expensive
- Could reduce registration accuracy

---

#### 3.3 Graph-Based Slice Ordering Optimization ⭐⭐
**Impact:** Medium-High | **Effort:** High | **Risk:** Medium

**Current State:**
- Fixed cascading order from anchor slice
- No optimization of registration order

**Recommendation:**
```python
def optimize_registration_order(slice_similarities):
    """
    Build graph where edges = similarity between slices.
    Find optimal spanning tree for registration order.
    Register most similar slices first, propagate outward.
    """
```

**Implementation:**
- Compute pairwise slice similarities (MI, correlation)
- Build minimum spanning tree
- Register along tree edges
- More robust to problem slices

**Benefits:**
- Optimal registration propagation
- Avoids error accumulation
- Handles missing/damaged slices better

**Risks:**
- Complex to implement
- May not integrate easily with current pipeline
- Requires pairwise comparisons (N^2 complexity)

---

## Recommended Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

**Goal:** Immediate improvement with minimal risk

1. ✅ **Enable final stack smoothing** (1.1)
   - Add `final_stack_smoothing_sigma = 2` parameter
   - Apply Gaussian smoothing in Z-direction to final output
   - **Expected improvement:** 30-40% reduction in jaggedness

2. ✅ **Increase SyN smoothing parameters** (1.2)
   - `syn_flow_sigma = 4` (from 3)
   - `syn_total_sigma = 2` (from 0)
   - **Expected improvement:** 20-30% smoother deformations

3. ✅ **Test and validate**
   - Run on subset of data
   - Visual inspection of results
   - Compute metrics (if available)

### Phase 2: Deformation Field Processing (3-5 days)

**Goal:** Enforce smoothness at the transform level

1. ✅ **Implement deformation field smoothing** (1.3)
   - Create `smooth_deformation_fields_across_stack()` function
   - Integrate into pipeline after registration
   - **Expected improvement:** 25-35% reduction in discontinuities

2. ✅ **Add median filtering for outliers** (2.3)
   - Quick implementation
   - Removes registration spikes
   - **Expected improvement:** 10-15% reduction in artifacts

3. ✅ **Test and validate**
   - Compare with Phase 1 results
   - Ensure no loss of alignment quality

### Phase 3: Advanced Smoothing (5-7 days)

**Goal:** Sophisticated edge-preserving smoothing

1. ✅ **Implement bilateral filtering** (2.2)
   - Z-direction only
   - Edge-preserving smoothness
   - **Expected improvement:** 15-20% better preservation of boundaries

2. ✅ **Adaptive regularization** (2.1)
   - MI-based regularization adjustment
   - Targeted smoothing for problem slices
   - **Expected improvement:** 10-20% improvement in difficult regions

3. ✅ **Comprehensive testing**
   - Full dataset validation
   - Quality metrics
   - Visual assessment

### Phase 4: Advanced Approaches (If Needed - 2+ weeks)

**Goal:** Fundamental improvements if simpler methods insufficient

1. 🔄 **Sliding window registration** (3.1)
   - Major refactoring
   - Test on pilot data first
   - **Expected improvement:** 40-50% reduction in jaggedness

2. 🔄 **Global 3D regularization** (3.2)
   - Research implementation approaches
   - May require external libraries
   - **Expected improvement:** Optimal smoothness

## Parameter Tuning Guidelines

### Smoothing Parameters

| Parameter | Current | Conservative | Moderate | Aggressive |
|-----------|---------|--------------|----------|------------|
| `final_stack_smoothing_sigma` | 0 | 1-2 | 2-3 | 3-5 |
| `syn_flow_sigma` | 3 | 4 | 5 | 6 |
| `syn_total_sigma` | 0 | 1 | 2 | 3 |
| `deformation_field_smooth_sigma` | N/A | 0.5-1 | 1-2 | 2-3 |

### Tuning Process

1. **Start conservative** - Use lower smoothing values first
2. **Visual inspection** - Check for jaggedness vs blur tradeoff
3. **Incremental increases** - Increase by 0.5-1.0 units at a time
4. **Validate alignment** - Ensure MI/correlation doesn't degrade
5. **Document results** - Record parameter values and outcomes

### Quality Metrics to Track

- **Jaggedness score**: Standard deviation of slice-to-slice differences
- **Continuity metric**: Gradient magnitude in Z-direction
- **Alignment quality**: MI between adjacent registered slices
- **Visual assessment**: Expert review of selected regions

## Implementation Checklist

### Phase 1: Quick Wins

- [ ] Add `final_stack_smoothing_sigma` parameter to run_slice_registration.py
- [ ] Implement final stack smoothing in generate_stack_and_template()
- [ ] Increase `syn_flow_sigma` to 4
- [ ] Enable `syn_total_sigma` = 2
- [ ] Test on subset (10-20 slices)
- [ ] Visual comparison before/after
- [ ] Document results

### Phase 2: Deformation Field Processing

- [ ] Implement `smooth_deformation_fields_across_stack()` function
- [ ] Add integration point in registration pipeline
- [ ] Implement median filtering for deformation fields
- [ ] Test on full dataset
- [ ] Measure improvement metrics
- [ ] Document results

### Phase 3: Advanced Smoothing

- [ ] Implement bilateral filtering in Z-direction
- [ ] Implement adaptive regularization based on MI
- [ ] Integrate into pipeline
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Final documentation

## Expected Outcomes

### Cumulative Improvement Estimates

After **Phase 1** (Quick Wins):
- **40-60%** reduction in visual jaggedness
- **Minimal** impact on alignment quality
- **Low** computational overhead

After **Phase 2** (Deformation Field Processing):
- **60-75%** reduction in discontinuities
- **Better** preservation of registration quality
- **Moderate** computational overhead

After **Phase 3** (Advanced Smoothing):
- **75-85%** reduction in jaggedness
- **Optimal** balance of smoothness vs detail
- **Acceptable** computational overhead

### Success Criteria

✅ **Minimum Acceptable:**
- Visibly smoother slice-to-slice transitions
- No significant loss in alignment quality
- Acceptable computation time increase (<50%)

✅ **Target:**
- Jaggedness reduced to near-imperceptible levels
- Maintained or improved alignment quality
- Computation time increase <25%

✅ **Stretch Goal:**
- Publication-quality 3D reconstructions
- Automated quality metrics show consistent improvement
- Generalizes across different datasets

## Risks and Mitigation

### Risk 1: Over-Smoothing
**Problem:** Too much smoothing blurs important features
**Mitigation:** 
- Start with conservative parameters
- Validate preservation of boundaries
- Use edge-preserving methods (bilateral filtering)

### Risk 2: Reduced Alignment Quality
**Problem:** Smoothing constraints may reduce registration accuracy
**Mitigation:**
- Track MI scores throughout
- Compare registered vs unregistered images
- Roll back if quality degrades significantly

### Risk 3: Increased Computation Time
**Problem:** Additional processing may slow pipeline significantly
**Mitigation:**
- Profile performance bottlenecks
- Parallelize where possible
- Make smoothing optional/configurable

### Risk 4: Parameter Sensitivity
**Problem:** Results may be highly sensitive to parameter choices
**Mitigation:**
- Document parameter ranges thoroughly
- Provide sensible defaults
- Implement parameter sweep utility

## Maintenance and Documentation

### Code Documentation Needs

1. **Parameter descriptions** - What each smoothing parameter does
2. **Tuning guide** - How to adjust for different datasets
3. **Quality metrics** - How to measure improvement
4. **Troubleshooting** - Common issues and solutions

### User Documentation Needs

1. **Quick start guide** - Enabling smoothing with defaults
2. **Parameter reference** - Complete parameter documentation
3. **Case studies** - Before/after examples with different parameters
4. **Best practices** - Recommendations for different scenarios

## References and Resources

### Relevant Literature

1. **Deformation field smoothing:**
   - Beg et al. (2005) - "Computing large deformation metric mappings via geodesic flows"
   - Smoothness constraints in diffeomorphic registration

2. **Multi-slice registration:**
   - Yushkevich et al. (2006) - "Deformable registration of diffusion tensor MR images"
   - Slice-to-volume registration approaches

3. **Regularization strategies:**
   - Modersitzki (2004) - "Numerical Methods for Image Registration"
   - Theory of regularization in image registration

### Existing Implementations

1. **ANTs SyN parameters:**
   - Official ANTs documentation on flow_sigma and total_sigma
   - Best practices for different image modalities

2. **SimpleITK smoothing:**
   - Deformation field smoothing examples
   - Gaussian and bilateral filtering APIs

## Conclusion

The jaggedness in slice registration results stems from **independent per-slice registrations without explicit smoothness constraints across the Z-axis**. The most effective improvements will come from:

1. ✅ **Enabling final output smoothing** (currently disabled)
2. ✅ **Increasing deformation field smoothing** (currently minimal)
3. ✅ **Post-processing transform smoothness** (currently absent)

The recommended **phased approach** allows for incremental improvements with manageable risk, starting with quick wins that can be validated before proceeding to more complex solutions.

**Estimated timeline for significant improvement: 1-2 weeks** (Phases 1-2)
**Estimated timeline for optimal results: 3-4 weeks** (Phases 1-3)

The plan addresses the identified TODOs in the codebase while providing a clear path to publication-quality 3D reconstructions.
