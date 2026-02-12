"""
Simple validation script for resolution support implementation.

This script validates that:
1. Both Python files have no syntax errors
2. The key changes are present in the code
3. Function signatures include the use_resolution parameter
"""

import re
import sys


def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    print(f"\nChecking syntax of {filepath}...")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        print(f"✓ {filepath} has valid Python syntax")
        return True, code
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False, None


def check_parameter_in_function(code, function_name, parameter_name):
    """Check if a function definition includes a specific parameter."""
    # Pattern to match function definition with the parameter
    pattern = rf'def {function_name}\([^)]*{parameter_name}'
    if re.search(pattern, code):
        print(f"  ✓ {function_name} includes '{parameter_name}' parameter")
        return True
    else:
        print(f"  ✗ {function_name} missing '{parameter_name}' parameter")
        return False


def check_variable_in_code(code, variable_name):
    """Check if a variable is defined in the code."""
    pattern = rf'^{variable_name}\s*='
    if re.search(pattern, code, re.MULTILINE):
        print(f"  ✓ Variable '{variable_name}' is defined")
        return True
    else:
        print(f"  ✗ Variable '{variable_name}' not found")
        return False


def check_ignore_res_usage(code):
    """Check if ignore_res is properly controlled."""
    # Should use "not use_resolution" to invert the logic
    pattern = r'ignore_res\s*=\s*not\s+use_resolution'
    if re.search(pattern, code):
        print(f"  ✓ ignore_res properly controlled by use_resolution parameter")
        return True
    else:
        print(f"  ✗ ignore_res not properly controlled")
        return False


def check_spacing_reset(code):
    """Check if spacing is reset in do_reg_ants when not using resolution."""
    pattern = r'if\s+not\s+use_resolution:.*?set_spacing\(\(1\.0,\s*1\.0,\s*1\.0\)\)'
    if re.search(pattern, code, re.DOTALL):
        print(f"  ✓ Spacing reset logic present in do_reg_ants")
        return True
    else:
        print(f"  ✗ Spacing reset logic not found")
        return False


def validate_run_slice_registration(code):
    """Validate changes in run_slice_registration.py."""
    print("\n=== Validating run_slice_registration.py ===")
    
    checks = [
        check_variable_in_code(code, 'use_resolution_in_registration'),
        check_variable_in_code(code, 'in_plane_res_x'),
        check_variable_in_code(code, 'in_plane_res_y'),
        check_variable_in_code(code, 'in_plane_res_z'),
    ]
    
    # Check if resolution is conditionally set in affine
    if 'if use_resolution_in_registration:' in code:
        print("  ✓ Conditional resolution setting in affine matrix")
        checks.append(True)
    else:
        print("  ✗ Conditional resolution setting not found")
        checks.append(False)
    
    # Check if use_resolution parameter is passed to functions
    if 'use_resolution=use_resolution_in_registration' in code:
        print("  ✓ use_resolution parameter passed to registration functions")
        checks.append(True)
    else:
        print("  ✗ use_resolution parameter not passed to functions")
        checks.append(False)
    
    return all(checks)


def validate_slice_registration_functions(code):
    """Validate changes in slice_registration_functions.py."""
    print("\n=== Validating slice_registration_functions.py ===")
    
    checks = [
        check_parameter_in_function(code, 'do_reg', 'use_resolution'),
        check_parameter_in_function(code, 'do_reg_ants', 'use_resolution'),
        check_parameter_in_function(code, 'coreg_single_slice_orig', 'use_resolution'),
        check_parameter_in_function(code, 'run_parallel_coregistrations', 'use_resolution'),
        check_parameter_in_function(code, 'run_cascading_coregistrations_v2', 'use_resolution'),
        check_ignore_res_usage(code),
        check_spacing_reset(code),
    ]
    
    return all(checks)


def main():
    """Main validation function."""
    print("=" * 70)
    print("Resolution Support Implementation Validation")
    print("=" * 70)
    
    # Check run_slice_registration.py
    success1, code1 = check_file_syntax('run_slice_registration.py')
    if success1:
        result1 = validate_run_slice_registration(code1)
    else:
        result1 = False
    
    # Check slice_registration_functions.py
    success2, code2 = check_file_syntax('slice_registration_functions.py')
    if success2:
        result2 = validate_slice_registration_functions(code2)
    else:
        result2 = False
    
    # Final summary
    print("\n" + "=" * 70)
    if result1 and result2:
        print("✓ All validation checks passed!")
        print("=" * 70)
        print("\nThe resolution support has been successfully implemented:")
        print("  • Both files have valid Python syntax")
        print("  • use_resolution_in_registration variable added")
        print("  • All registration functions accept use_resolution parameter")
        print("  • Resolution is conditionally set in NIfTI creation")
        print("  • ignore_res is properly controlled in do_reg")
        print("  • Spacing is reset in do_reg_ants when needed")
        print("  • Parameters are passed through the call chain")
        print("\nTo enable resolution, set: use_resolution_in_registration = True")
        print("in run_slice_registration.py (line ~40)")
        return 0
    else:
        print("✗ Some validation checks failed")
        print("=" * 70)
        print("\nPlease review the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
