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
    H_html_file_name, H_pkl_file_name, hopping_parameters_template_file_name,representations_all_file_name

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
    conf_file_dir = Path(parsed_config.get('config_file_path', 'N/A')).parent
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
# print(sgr_result.stdout)
print(f"Exit code: {sgr_result.returncode}")
# Check if space group representations were generated successfully
if sgr_result.returncode != 0:
    print("Space group representations generation failed!")
    print(f"return code={sgr_result.returncode}")
    print("Error output:")
    print(sgr_result.stderr)
    print("Standard output:")
    print(sgr_result.stdout)
    exit(sgr_result.returncode)


else:
    print("Space group representations generated successfully!")
    # Parse the JSON output
    pkl_file_not_found_err_code = 40
    pkl_unpickling_err_code = 41
    pkl_general_err_code = 42
    try:
        representations_all_file_name_path = Path(conf_file_dir) / representations_all_file_name
        with open(representations_all_file_name_path,"rb") as fptr:
            magnetic_space_group_representations = pickle.load(fptr)
        print(f"Successfully loaded representations from {representations_all_file_name_path}")

    except FileNotFoundError:
        print(f"Error: Could not find the representations file at {representations_all_file_name_path}")
        exit(pkl_file_not_found_err_code)
    except pickle.UnpicklingError:
        print(f"Error: The file {representations_all_file_name_path} is corrupted or not a valid pickle file.")
        exit(pkl_unpickling_err_code)

    except Exception as e:
        print(f"An unexpected error occurred while loading the representations: {e}")
        exit(pkl_general_err_code)

lattice_basis = np.array(parsed_config['lattice_basis'])
print("\n" + "=" * 60)
print("COMPLETING ORBITALS UNDER SYMMETRY")
print("=" * 60)

# Convert to JSON for subprocess
parsed_config_json=json.dumps(parsed_config)
# Run complete_orbitals.py
completing_result = subprocess.run(
    ["python3", "./symmetry/complete_orbitals.py"],
    input=parsed_config_json,
    capture_output=True,
    text=True
)

# print(completing_result.stdout)
# Check if orbital completion succeeded
if completing_result.returncode != 0:
    print("Orbital completion failed!")
    print(f"Return code: {completing_result.returncode}")
    print("Error output:")
    print(completing_result.stderr)
    exit(completing_result.returncode)


# Parse the output
try:
    orbital_completion_data = json.loads(completing_result.stdout)
    print("Orbital completion successful!")
    # Display which orbitals were added by symmetry
    print("\n" + "-" * 40)
    print("ORBITALS ADDED BY SYMMETRY:")
    print("-" * 40)
    added_orbitals = orbital_completion_data["added_orbitals"]
    if any(added_orbitals.values()):
        for atom_type, orbitals in added_orbitals.items():
            if orbitals:
                print(f"  {atom_type}: {', '.join(orbitals)}")
    else:
        print("  No additional orbitals needed - input was already complete")

        # Display final active orbitals for each atom
    print("\n" + "-" * 40)
    print("FINAL ACTIVE ORBITALS PER ATOM:")
    print("-" * 40)
    updated_vectors = orbital_completion_data["updated_orbital_vectors"]

    orbital_map_reverse = {v: k for k, v in orbital_map.items()}  # Reverse lookup
    for atom_type, vector in updated_vectors.items():
        # Find indices where orbital is active (value = 1)
        active_indices = [i for i, val in enumerate(vector) if val == 1]
        # Convert indices back to orbital names
        active_orbital_names = [orbital_map_reverse.get(idx, f"unknown_{idx}") for idx in active_indices]
        print(f"  {atom_type} ({len(active_orbital_names)} orbitals): {', '.join(active_orbital_names)}")
    # Display symmetry representation information
    print("\n" + "-" * 40)
    print("SYMMETRY REPRESENTATIONS ON ACTIVE ORBITALS:")
    print("-" * 40)
    representations = orbital_completion_data["representations_on_active_orbitals"]
    for atom_type, repr_matrices in representations.items():
        if repr_matrices:
            repr_array = np.array(repr_matrices)
            print(
                f"  {atom_type}: {repr_array.shape[0]} operations, {repr_array.shape[1]}×{repr_array.shape[2]} matrices")

    # Update parsed_config with completed orbitals
    for atom_pos in parsed_config['Wyckoff_positions']:
        position_name = atom_pos['position_name']
        print(f"Updating orbitals for position_name={position_name}")
        # Get the updated orbital vector for this atom
        if position_name in updated_vectors:
            vector = updated_vectors[position_name]
            active_indices = [i for i, val in enumerate(vector) if val == 1]
            active_orbital_names = [orbital_map_reverse.get(idx) for idx in active_indices]
            # Update the specific position entry (consistency)
            atom_pos['orbitals'] = active_orbital_names
            # Update the Wyckoff_position_types dictionary for this position_name
            # CHANGED: maintain dictionary structure
            parsed_config['Wyckoff_position_types'][position_name] = {'orbitals': active_orbital_names}

    # Store completion results for later use
    orbital_completion_results = {
        "status": "completed",
        "added_orbitals": added_orbitals,
        "orbital_vectors": updated_vectors,
        "representations_on_active_orbitals": representations,
    }

except json.JSONDecodeError as e:
    print("Error parsing JSON output from complete_orbitals.py:")
    print(f"JSON Error: {e}")
    print("Raw output:")
    print(completing_result.stdout)
    print("Error output:")
    print(completing_result.stderr)
    exit(1)

except KeyError as e:
    print(f"Missing key in orbital completion output: {e}")
    print("Available keys:",
          list(orbital_completion_data.keys()) if 'orbital_completion_data' in locals() else "Could not parse JSON")
    exit(1)


except Exception as e:
    print(f"Unexpected error processing orbital completion: {e}")
    print("Type:", type(e).__name__)
    exit(1)

print("\n" + "=" * 60)
print("ORBITAL COMPLETION FINISHED")
print("=" * 60)
print(f"parsed_config['Wyckoff_position_types']={parsed_config['Wyckoff_position_types']}")
print(f"parsed_config['Wyckoff_positions']={parsed_config['Wyckoff_positions']}")


# ==============================================================================
# Save preprocessing data to pickle file
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING PREPROCESSING DATA")
print("=" * 80)
# Prepare comprehensive preprocessing data package
origin_cart = [0, 0, 0]  # origin for .cif file
origin_cart = np.array(origin_cart)
repr_s, repr_p, repr_d, repr_f = magnetic_space_group_representations["repr_s_p_d_f"]
repr_s_np = np.array(repr_s)
repr_p_np = np.array(repr_p)
repr_d_np = np.array(repr_d)
repr_f_np = np.array(repr_f)

magnetic_space_group_matrices_cartesian= np.array(magnetic_space_group_representations["magnetic_space_group_matrices_spatial_cartesian"])
magnetic_space_group_cart =[np.array(item) for item in magnetic_space_group_matrices_cartesian]
print(f"directions_to_study={directions_to_study}")
search_dim = parsed_config['dim']  # =len(directions_to_study)
# print(f"search_dim={search_dim}")
spinor_mat_representation=np.array(magnetic_space_group_representations["spinor_mat_representation"])

delta_vec=np.array(magnetic_space_group_representations["delta_vec"])

preprocessing_data = {
    # Core configuration
    'parsed_config': parsed_config,
    "name": parsed_config["name"],
    # magnetic Space group representations
    'magnetic_space_group_representations': magnetic_space_group_representations,
    'directions_to_study':directions_to_study,
    "dim":search_dim,
    # NumPy arrays for efficient computation
    'magnetic_space_group_cart': magnetic_space_group_cart,  # List of np.ndarray
    "spinor_mat_representation":spinor_mat_representation,
    "delta_vec":delta_vec,
    'origin_cart': origin_cart,  # np.ndarray (3,)
    # Orbital representation matrices
    'repr_s_np': repr_s_np,  # np.ndarray (num_ops, 1, 1)
    'repr_p_np': repr_p_np,  # np.ndarray (num_ops, 3, 3)
    'repr_d_np': repr_d_np,  # np.ndarray (num_ops, 5, 5)
    'repr_f_np': repr_f_np,  # np.ndarray (num_ops, 7, 7)
    # Orbital completion results
    'orbital_completion_results': orbital_completion_results,
    # Orbital mapping dictionary
    'orbital_map': orbital_map,
    # Metadata
    'creation_date': datetime.now().isoformat(),
    'script_version': '1.0',
    'description': 'Preprocessing data for tight-binding model construction'
}
# Determine output file path
config_file_path = parsed_config["config_file_path"]
config_dir = Path(config_file_path).parent
preprocessed_pickle_file =  str(config_dir/processed_input_pkl_file_name)
# Save to pickle file
try:
    with open(preprocessed_pickle_file, 'wb') as f:
        pickle.dump(preprocessing_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Calculate file size
    file_size = Path(preprocessed_pickle_file).stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024 ** 2:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024 ** 2):.2f} MB"
    print(f"✓ Preprocessing data saved successfully!")
    print(f"  File: {preprocessed_pickle_file}")
    print(f"  Size: {size_str}")
    print(f"\nSaved data includes:")
    print(f"  - parsed_config: Configuration dictionary")
    print(f"  - magnetic_space_group_representations: Full representation data")
    print(f"  - magnetic_space_group_cart: {len(magnetic_space_group_cart)} operations")
    print(f"  - origin_cart: magnetic Space group origin")
    print(f"  - repr_s/p/d/f_np: Orbital representation matrices")
    print(f"  - orbital_completion_results: Symmetry-completed orbitals")
    print(f"  - orbital_map: 78-dimensional orbital mapping")

except Exception as e:
    print(f"✗ Failed to save preprocessing data!")
    print(f"  Error: {e}")
    exit(save_err_code)

print("=" * 80)
