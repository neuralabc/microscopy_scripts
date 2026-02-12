# 2D Slice Registration - Modular Structure

This directory contains the 2D slice registration pipeline split into two files for better maintainability:

## Files

### `slice_registration_functions.py`
Contains all reusable function definitions for the slice registration pipeline:
- Image preprocessing functions (histogram matching, downsampling, etc.)
- Registration functions (rigid, non-linear, cascading registrations)
- Template generation functions
- Utility functions (logging, file operations, etc.)

This module can be imported and reused in different scripts or contexts.

### `run_slice_registration.py`
Application script that applies the registration functions to a specific dataset:
- Imports all functions from `slice_registration_functions.py`
- Contains dataset-specific configuration parameters (subject, paths, rescale values, etc.)
- Executes the registration pipeline on the configured dataset

## Original File

The original monolithic script `2d_slice_registration_all_fns_parallel_HPC_regTests_v7_cascadetest_res.py` 
remains available for reference but is now superseded by the modular structure.

## Usage

To run the registration pipeline:

```python
python run_slice_registration.py
```

To use the functions in other scripts:

```python
from slice_registration_functions import *

# Now you can use any of the defined functions
template = generate_stack_and_template(output_dir, subject, all_image_fnames)
```

## Function Summary

The `slice_registration_functions.py` module contains 28 functions and 1 class:
- `compute_histogram_matched_slice` - Histogram matching between slices
- `generate_gaussian_weights` - Weight generation for neighboring slices
- `do_reg_ants` - ANTs-based registration
- `run_cascading_coregistrations_v2` - Main cascading registration workflow
- `generate_stack_and_template` - Stack creation and template generation
- `setup_logging` - Logging configuration
- And 22 more specialized functions

See the function file for detailed documentation of each function.
