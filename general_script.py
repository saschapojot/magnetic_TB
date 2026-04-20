import subprocess
import sys
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import sympy as sp
import pickle
import base64

from name_conventions import orbital_map, processed_input_pkl_file_name, type_linear, type_hermitian, H_latex_file_name, \
    H_html_file_name, H_pkl_file_name, hopping_parameters_template_file_name

sp.init_printing(use_unicode=False, wrap_line=False)
# this script computes for magnetic space group system

# ==============================================================================
# STEP 1: Validate command line arguments
# ==============================================================================
argErrCode = 20
save_err_code = 30
json_err_code = 31
json_err_code_2 = 32

if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python general_script.py /path/to/xxx.conf")
    exit(argErrCode)

confFileName = str(sys.argv[1])
# ==============================================================================
# STEP 2: Parse configuration file
# ==============================================================================
# Run parse_conf.py to read and parse the configuration file
confResult = subprocess.run(
    ["python3", "./parse_files/parse_conf.py", confFileName],
    capture_output=True,
    text=True
)

# Check if the subprocess ran successfully
if confResult.returncode != 0:
    print("Error running parse_conf.py:")
    print(confResult.stderr)
    exit(confResult.returncode)

# Parse the JSON output from parse_conf.py
try:
    parsed_config = json.loads(confResult.stdout)
    # Display parsed configuration in a formatted way
    print("=" * 60)
    print("COMPLETE PARSED CONFIGURATION")
    print("=" * 60)
    # 1. General System Information
    print(f"{'System Name:':<25} {parsed_config.get('name', 'N/A')}")
    print(f"{'Config File:':<25} {parsed_config.get('config_file_path', 'N/A')}")
    print(f"{'Dimension:':<25} {parsed_config.get('dim', 'N/A')}")
    directions_to_study = parsed_config.get('directions_to_study')
    directions_str = ", ".join(directions_to_study) if directions_to_study else "None"
    print(f"{'Directions to study:':<25} {directions_str}")

    print(f"{'Spin considered:':<25} {parsed_config.get('spin', 'N/A')}")
    print(f"{'Truncation Radius:':<25} {parsed_config.get('truncation_radius', 'N/A')}")

    print("-" * 60)
    # 2. Space Group & Lattice Information
    print(f"{'Space Group number:':<25} {parsed_config.get('space_group', 'N/A')}")
    print(f"{'H-M Name:':<25} {parsed_config.get('space_group_name_H_M', 'N/A')}")
    print(f"{'Cell Setting:':<25} {parsed_config.get('cell_setting', 'N/A')}")
    origin = parsed_config.get('space_group_origin')
    origin_str = f"{origin}" if origin else "N/A"
    print(f"{'Space Group Origin:':<25} {origin_str}")

    print("-" * 60)
    # 2.5 Magnetic Space Group Information
    print("Magnetic Space Group Information:")
    print(f"{'BNS Number:':<25} {parsed_config.get('space_group_magn_number_BNS', 'N/A')}")
    print(f"{'UNI Name:':<25} {parsed_config.get('space_group_magn_name_UNI', 'N/A')}")
    print(f"{'BNS Name:':<25} {parsed_config.get('space_group_magn_name_BNS', 'N/A')}")
    print(f"{'OG Number:':<25} {parsed_config.get('space_group_magn_number_OG', 'N/A')}")
    print(f"{'OG Name:':<25} {parsed_config.get('space_group_magn_name_OG', 'N/A')}")
    print(f"{'Litvin PG Number:':<25} {parsed_config.get('point_group_number_Litvin', 'N/A')}")
    print(f"{'UNI PG Name:':<25} {parsed_config.get('point_group_name_UNI', 'N/A')}")

    print("\nLattice Basis Vectors:")
    basis = parsed_config.get('lattice_basis')
    if basis and isinstance(basis, list):
        for i, vec in enumerate(basis):
            print(f"  Vector {i + 1}: {vec}")
    else:
        print("  N/A")

    print("-" * 60)
    # 3. Wyckoff Positions & Orbitals
    print(f"{'Wyckoff position number:':<25} {parsed_config.get('Wyckoff_position_num', 'N/A')}")

    print("\nAtom/Orbital Definitions:")
    atom_types = parsed_config.get('Wyckoff_position_types', {})
    if atom_types:
        for atom, data in atom_types.items():
            # CHANGED: handle dictionary structure {'orbitals': [...]}
            orbitals = data.get('orbitals', []) if isinstance(data, dict) else data
            print(f"  {atom:<5} : {', '.join(orbitals)}")
    else:
        print("  No atoms defined.")
    # ---------------------------------------------------------
    # NEW SECTION: Print Position Coefficients
    # ---------------------------------------------------------
    print("\nAtom Position Coefficients:")
    wyckoff_positions = parsed_config.get('Wyckoff_positions', [])
    if wyckoff_positions:
        # Sort by position_name for cleaner output
        wyckoff_positions.sort(key=lambda x: x.get('position_name', ''))

        for pos in wyckoff_positions:
            # CHANGED: 'label' -> 'position_name'
            position_name = pos.get('position_name', 'Unknown')
            # CHANGED: 'position' -> 'fractional_coordinates'
            coords = pos.get('fractional_coordinates')

            if coords:
                # Format coordinates nicely (e.g., [0.333, 0.667, 0.0])
                coords_str = f"[{coords[0]}, {coords[1]}, {coords[2]}]"
                print(f"  {position_name:<5} : {coords_str}")
            else:
                print(f"  {position_name:<5} : No coordinates defined")
    else:
        print("  No positions found.")
    # ---------------------------------------------------------
    print("=" * 60)






except json.JSONDecodeError as e:
    print("Failed to parse JSON output from parse_conf.py")
    print(f"Error: {e}")
    print("Raw output:")
    print(confResult.stdout)
    exit(json_err_code)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(json_err_code_2)


# Convert parsed_config to JSON string for passing to other subprocesses
config_json = json.dumps(parsed_config)

# ==============================================================================
# STEP 3: Run sanity checks on parsed configuration
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING SANITY CHECK")
print("=" * 60)

# Run sanity_check.py and pass the JSON data via stdin
sanity_result = subprocess.run(
    ["python3", "./parse_files/sanity_check.py"],
    input=config_json,
    capture_output=True,
    text=True
)
print(f"Exit code: {sanity_result.returncode}")


# ==============================================================================
# STEP 4: Generate magnetic space group representations
# ==============================================================================
print("\n" + "=" * 60)
print("COMPUTING SPACE GROUP REPRESENTATIONS")
print("=" * 60)
sgr_result = subprocess.run(
    ["python3", "./symmetry/generate_magnetic_space_group_representations.py"],
    input=config_json,
    capture_output=True,
    text=True
)
print(sgr_result.stdout)
print(f"Exit code: {sgr_result.returncode}")
