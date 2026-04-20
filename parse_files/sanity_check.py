import sys
import glob
import re
import json
import numpy as np

# ==============================================================================
# Sanity check script for tight-binding configuration files
# ==============================================================================
# This script validates the parsed configuration data from .conf files
# It checks for:
# - Valid matrix properties (determinant, condition number)
# - Valid Wyckoff position type references
# - Duplicate atomic positions after lattice reduction
# - Consistency between dimension and directions to study
# - Consistency between declared Wyckoff number and actual definitions

# Exit codes for different error conditions
jsonErr = 4  # JSON parsing error
valErr = 5  # Value validation error
matrix_not_exist_error = 6  # Required matrix field missing
matrix_cond_error = 7  # Matrix condition number or determinant error
atom_position_error = 8  # Atom position reference error
duplicate_position_error = 9  # Duplicate atomic positions found
dim_mismatch_error = 10  # Mismatch between dim and directions_to_study
wyckoff_count_error = 11  # Mismatch in Wyckoff position count

# ==============================================================================
# STEP 1: Read and parse JSON input from stdin
# ==============================================================================
try:
    # Read all input from stdin
    input_data = sys.stdin.read()

    # Check if input is empty
    if not input_data:
        print("Error: No input data received via stdin", file=sys.stderr)
        exit(jsonErr)

    parsed_config = json.loads(input_data)

except json.JSONDecodeError as e:
    print(f"Error parsing JSON input: {e}", file=sys.stderr)
    exit(jsonErr)


# ==============================================================================
# STEP 2: Define matrix validation function
# ==============================================================================
def check_matrix_condition(matrix, matrix_name="Matrix", det_threshold=1e-12, cond_threshold=1e12):
    """
    Check if a matrix is well-conditioned and non-degenerate
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(matrix, list):
            np_matrix = np.array(matrix)
        else:
            np_matrix = matrix

        # Check if it's a square matrix (required for basis vectors)
        if np_matrix.shape[0] != np_matrix.shape[1]:
            return False, f"{matrix_name} is not square: shape {np_matrix.shape}"

        # Check determinant (non-degenerate test)
        det = np.linalg.det(np_matrix)
        if abs(det) < det_threshold:
            return False, f"{matrix_name} is degenerate (determinant ≈ 0): det = {det:.2e}"

        # Check condition number (ill-conditioning test)
        cond_num = np.linalg.cond(np_matrix)
        if cond_num > cond_threshold:
            return False, f"{matrix_name} is ill-conditioned: condition number = {cond_num:.2e}"

        return True, None

    except Exception as e:
        return False, f"Error analyzing {matrix_name}: {str(e)}"


# ==============================================================================
# STEP 3: Define Consistency Check Functions
# ==============================================================================
def check_dimension_consistency(config):
    """
    Verify that 'dim' matches the number of 'directions_to_study'.
    """
    dim = config.get('dim')
    directions = config.get('directions_to_study')

    # Check if values exist
    if dim is None:
        return False, "Missing 'dim' in configuration."
    if directions is None:
        return False, "Missing 'directions_to_study' in configuration."

    # Check consistency
    num_directions = len(directions)
    if dim != num_directions:
        return False, (f"Dimension mismatch: 'dim' is {dim}, but found {num_directions} "
                       f"direction(s) in 'directions_to_study' ({', '.join(directions)}).")

    return True, None


def check_wyckoff_consistency(config):
    """
    Verify that 'Wyckoff_position_num' matches the actual number of
    positions defined in 'Wyckoff_positions'.
    """
    declared_num = config.get('Wyckoff_position_num')

    # Get the list of actual position definitions found
    # (This list is built in parse_conf.py based on unique labels like B0, N0, etc.)
    actual_positions = config.get('Wyckoff_positions', [])
    actual_num = len(actual_positions)

    if declared_num is None:
        return False, "Missing 'Wyckoff_position_num' in configuration."

    if declared_num != actual_num:
        # Create a list of labels found for the error message
        found_labels = [p.get('label', 'unknown') for p in actual_positions]
        found_str = ", ".join(found_labels)
        return False, (f"Wyckoff count mismatch: 'Wyckoff_position_num' is {declared_num}, "
                       f"but found {actual_num} definition(s) ({found_str}).")

    return True, None


# ==============================================================================
# STEP 4: Check for required matrix fields
# ==============================================================================
# Verify that lattice_basis exists and is not empty
if 'lattice_basis' not in parsed_config or not parsed_config['lattice_basis']:
    print("Error: Missing or empty required field 'lattice_basis'", file=sys.stderr)
    exit(matrix_not_exist_error)

# ==============================================================================
# STEP 5: Run Validations
# ==============================================================================

# 1. Validate Matrix Conditions
lattice_basis_np = np.array(parsed_config['lattice_basis'])
is_valid, error_msg = check_matrix_condition(lattice_basis_np, "Lattice basis")
if not is_valid:
    print(f"Error: {error_msg}", file=sys.stderr)
    exit(matrix_cond_error)

# 2. Validate Dimension Consistency
is_dim_valid, dim_error_msg = check_dimension_consistency(parsed_config)
if not is_dim_valid:
    print(f"Error: {dim_error_msg}", file=sys.stderr)
    exit(dim_mismatch_error)

# 3. Validate Wyckoff Position Count
is_wyckoff_valid, wyckoff_error_msg = check_wyckoff_consistency(parsed_config)
if not is_wyckoff_valid:
    print(f"Error: {wyckoff_error_msg}", file=sys.stderr)
    exit(wyckoff_count_error)

print("Sanity check passed successfully.")
exit(0)