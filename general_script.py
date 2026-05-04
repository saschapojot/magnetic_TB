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

from classes.class_defs import frac_to_cartesian, atomIndex,hopping, vertex
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

magnetic_space_group_matrices_spatial_cartesian= np.array(magnetic_space_group_representations["magnetic_space_group_matrices_spatial_cartesian"])
#magnetic space group are represented in 3 parts, magnetic_space_group_cart_spatial is the spatial part
#spinor_mat_representation is the spinor part,
#delta_vec indicates time reversal
#spatial part
magnetic_space_group_cart_spatial =[np.array(item) for item in magnetic_space_group_matrices_spatial_cartesian]
print(f"directions_to_study={directions_to_study}")
search_dim = parsed_config['dim']  # =len(directions_to_study)
# print(f"search_dim={search_dim}")
#spinor part
spinor_mat_representation=np.array(magnetic_space_group_representations["spinor_mat_representation"])
#indicating time reversal
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
    'magnetic_space_group_cart_spatial': magnetic_space_group_cart_spatial,  # List of np.ndarray
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
    print(f"  - magnetic_space_group_cart_spatial: {len(magnetic_space_group_cart_spatial)} operations")
    print(f"  - origin_cart: magnetic Space group origin")
    print(f"  - repr_s/p/d/f_np: Orbital representation matrices")
    print(f"  - orbital_completion_results: Symmetry-completed orbitals")
    print(f"  - orbital_map: 78-dimensional orbital mapping")

except Exception as e:
    print(f"✗ Failed to save preprocessing data!")
    print(f"  Error: {e}")
    exit(save_err_code)

print("=" * 80)

def get_rotation_translation(magnetic_space_group_cart_spatial, operation_idx):
    """
    Extract rotation/reflection matrix R and translation vector t from a space group operation.

    The magnetic space group operation is in the form [R|t], represented as a 3×4 matrix:
        [R | t] = [R00 R01 R02 | t0]
                  [R10 R11 R12 | t1]
                  [R20 R21 R22 | t2]

    The operation transforms a position vector r as: r' = R @ r + t

    Args:
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
                                 using cif origin (shape: num_ops × 3 × 4)
        operation_idx: Index of the magnetic space group operation

    Returns:
        tuple: (R, t)
            - R (ndarray): 3×3 rotation/reflection matrix
            - t (ndarray): 3D translation vector
    """
    operation = magnetic_space_group_cart_spatial[operation_idx]
    R = operation[:3, :3]  # Rotation/reflection part
    t = operation[:3, 3]  # Translation part

    return R, t

def generate_wyckoff_orbit(wyckoff_position, magnetic_space_group_cart_spatial, lattice_basis,
                           tolerance=1e-3):
    """
    Generate all symmetry-equivalent positions (orbit) from a single Wyckoff position.
    Applies all magnetic space group operations' spatial parts to a Wyckoff position and collects unique
    atomic positions within the unit cell. This generates the complete orbit of
    the Wyckoff position under the magnetic space group.
    For each spatial operation [R|t], the transformation is:
        r' = R @ r + t
    where r is in fractional coordinates of the primitive cell.
    Positions that differ by a lattice vector are considered equivalent,
    so we reduce all positions to the range [0, 1) in fractional coordinates.
    Args:
        wyckoff_position: dict from parsed_config['Wyckoff_positions']
                         Must contain 'fractional_coordinates' key
                         Example: {'position_name': 'V0',
                                    'orbitals': ['3dxy', '3dyz', '3dxz']
                                  'fractional_coordinates':  [0.0, 0.5, 0.0]}
        magnetic_space_group_cart_spatial: List of spatial parts of magnetic space group matrices in Cartesian coordinates
                                using cif origin [0,0,0] (shape: num_ops × 3 × 4)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                        expressed in Cartesian coordinates using cif origin
        tolerance: Numerical tolerance for identifying duplicate positions (default: 1e-3)

    Returns: list of dicts: Each dict contains:
            - 'fractional_coordinates': [f0, f1, f2] in range [0, 1)
            - 'cartesian_coordinates': [x, y, z] in Cartesian coords (cif origin)
            - 'operation_idx': which space group operation generated this position
            - 'position_name': inherited from input Wyckoff position

    """
    # Extract input position in fractional coordinates
    r_frac_input = np.array(wyckoff_position['fractional_coordinates'])
    position_name = wyckoff_position['position_name']
    # Convert lattice basis to proper array and get transformation matrices
    lattice_basis = np.array(lattice_basis)  # rows are basis vectors
    lattice_matrix = np.column_stack(lattice_basis)  # Columns are basis vectors
    lattice_matrix_inv = np.linalg.inv(lattice_matrix)
    # Convert input fractional coordinates to Cartesian using cif origin
    r_cart_input = frac_to_cartesian([0, 0, 0], r_frac_input, lattice_basis, origin_cart)
    # Store unique positions
    unique_positions = []
    unique_frac_coords = []  # For deduplication
    # Apply each space group operation
    for op_idx, operation in enumerate(magnetic_space_group_cart_spatial):
        # Extract rotation and translation
        R, t = get_rotation_translation(magnetic_space_group_cart_spatial, op_idx)

        # Apply symmetry operation in Cartesian coordinates
        # r_cart' = R @ r_cart + t
        r_cart_transformed = R @ r_cart_input + t
        # Convert back to fractional coordinates
        r_frac_transformed = lattice_matrix_inv @ r_cart_transformed
        # Wrap to [0, 1) to stay within unit cell
        r_frac_wrapped = r_frac_transformed % 1.0
        # Check if this position is already in our list (within tolerance)
        is_duplicate = False
        for existing_frac in unique_frac_coords:
            # Check if positions are equivalent (accounting for periodic boundary conditions)
            diff = r_frac_wrapped - existing_frac
            if np.linalg.norm(diff) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            # Add to unique positions
            unique_frac_coords.append(r_frac_wrapped)

            # Convert wrapped fractional back to Cartesian for output
            r_cart_final = frac_to_cartesian([0, 0, 0], r_frac_wrapped, lattice_basis, origin_cart)

            # Create position dictionary
            position_dict = {
                'fractional_coordinates': r_frac_wrapped.tolist(),
                'cartesian_coordinates': r_cart_final.tolist(),
                'operation_idx': op_idx,
                'position_name': position_name,
            }
            unique_positions.append(position_dict)

    return unique_positions

def generate_atoms_in_unit_cell(parsed_config,magnetic_space_group_cart_spatial, lattice_basis,origin_cart,
                                repr_s, repr_p, repr_d, repr_f,
                                spinor_mat_representation,delta_vec,
                                tolerance=1e-3):
    """
    Generates all atoms in the unit cell by expanding the Wyckoff positions defined
    in the configuration using the provided magnetic space group operations.
    Args:
        parsed_config: Dictionary containing configuration (Wyckoff positions, origin, etc.)
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
        lattice_basis: 3x3 array of lattice basis vectors (each row is a basis vector)
        origin_cart: [0,0,0]
        repr_s, repr_p, repr_d, repr_f: Orbital representation matrices (numpy arrays)
        spinor_mat_representation: representation on spinors, time reversal considered
        delta_vec: delta values indicating whether a magnetic space group operation has time reversal,
                    1 means no time reversal, -1 means time reversal
        tolerance: Numerical tolerance for coordinate comparisons

    Returns: A list of atomIndex objects representing all atoms in the unit cell [0,0,0]

    """
    unit_cell_atoms = []
    # Iterate over all Wyckoff positions defined in the configuration
    for wyckoff_pos in parsed_config['Wyckoff_positions']:
        # Generate the full orbit (all equivalent atoms in unit cell)
        # This function applies magnetic space group spatial part operations to the Wyckoff generator
        orbit = generate_wyckoff_orbit(
            wyckoff_pos,
            magnetic_space_group_cart_spatial,
            lattice_basis,
            tolerance
        )
        # Create atomIndex objects for each position in the orbit
        for index, pos_data in enumerate(orbit):
            atom = atomIndex(
                cell=[0, 0, 0],  # Always the home unit cell
                frac_coord=pos_data['fractional_coordinates'],
                position_name=pos_data['position_name'],
                basis=lattice_basis,
                origin_cart=origin_cart,
                parsed_config=parsed_config,
                repr_s_np=repr_s,
                repr_p_np=repr_p,
                repr_d_np=repr_d,
                repr_f_np=repr_f,
                spinor_mat_representation=spinor_mat_representation,
                delta_vec=delta_vec
            )
            wyckoff_instance_id_tmp = atom.position_name + "_" + str(index)
            atom.wyckoff_instance_id = wyckoff_instance_id_tmp
            unit_cell_atoms.append(atom)
    return unit_cell_atoms



def compute_dist(center_atom, unit_cell_atoms, radius, search_range=10):
    """
    Find all atoms within a specified radius of a center atom by searching neighboring cells.
    Returns constructed atomIndex objects for all neighbors found. The neighboring atom types are determined by
    unit_cell_atoms.
    Args:
        center_atom:  atomIndex object for the center atom
        unit_cell_atoms:  list of atomIndex objects in the reference unit cell [0,0,0]
        radius:  cutoff distance in Cartesian coordinates (REQUIRED)
        search_range:  ow many cells to search in each direction (default: 10)

    Returns:
        list: atomIndex objects within the specified radius, sorted by distance

    """
    neighbor_atoms = []
    center_cart = center_atom.cart_coord
    lattice_basis = center_atom.basis
    origin_cart = center_atom.origin_cart
    # Define the full search span
    full_span = range(-search_range, search_range + 1)
    n0_range = full_span
    n1_range = full_span
    n2_range = full_span
    # Search through neighboring cells
    for n0 in n0_range:
        for n1 in n1_range:
            for n2 in n2_range:
                cell = [n0, n1, n2]
                # print(f"cell={cell}")
                # Check each atom in the unit cell
                for unit_atom in unit_cell_atoms:
                    # Compute Cartesian coordinates for this atom in the proposed cell
                    candidate_cart = frac_to_cartesian(cell, unit_atom.frac_coord, lattice_basis, origin_cart)
                    # print(f"candidate_cart={candidate_cart}")
                    # Calculate distance
                    dist = np.linalg.norm(candidate_cart - center_cart)
                    # print(f"dist={dist}")
                    # Only construct atom if it passes the distance check
                    if dist <= radius:
                        # Create atomIndex for this atom in the current cell with deep copies
                        # print("creating atom")
                        neighbor_atom = atomIndex(
                            cell=deepcopy(cell),
                            frac_coord=deepcopy(unit_atom.frac_coord),
                            position_name=unit_atom.position_name,
                            basis=deepcopy(lattice_basis),
                            origin_cart=deepcopy(origin_cart),
                            parsed_config=deepcopy(unit_atom.parsed_config),
                            repr_s_np=deepcopy(unit_atom.repr_s_np),
                            repr_p_np=deepcopy(unit_atom.repr_p_np),
                            repr_d_np=deepcopy(unit_atom.repr_d_np),
                            repr_f_np=deepcopy(unit_atom.repr_f_np),
                            spinor_mat_representation=deepcopy(unit_atom.spinor_mat_representation),
                            delta_vec=deepcopy(unit_atom.delta_vec)
                        )
                        neighbor_atom.wyckoff_instance_id = unit_atom.wyckoff_instance_id
                        neighbor_atoms.append((dist, neighbor_atom))
    # Sort by distance and return only the atomIndex objects
    neighbor_atoms.sort(key=lambda x: x[0])
    return [atom for dist, atom in neighbor_atoms]


def find_identity_operation(magnetic_space_group_cart_spatial, spinor_mat_representation, delta_vec, tolerance=1e-9):
    """
    Find the index of the identity operation in space group matrices.
    The identity operation has:
    - Rotation part: 3×3 identity matrix
    - Translation part: zero vector
    - spinor part: 2×2 identity matrix
    - delta: 1
    Args:
        magnetic_space_group_cart_spatial: List or array of  3×4 magnetic space group spatial part matrices [R|t]
                                 in Cartesian coordinates
        spinor_mat_representation: a list of spinor representations, each matrix is 2 × 2 unitary matrix
        delta_vec: a vector of 1 and -1, indicating time reversal
        tolerance: Numerical tolerance for comparison (default: 1e-9)

    Returns:
        int: Index of the identity operation
    Raises:
        ValueError: If identity operation is not found

    """
    identity_idx = None
    for idx in range(len(magnetic_space_group_cart_spatial)):
        # Extract rotation and translation using helper function
        R, t = get_rotation_translation(magnetic_space_group_cart_spatial, idx)

        # 1. Check if the rotation matrix is the 3x3 identity matrix
        is_R_identity = np.allclose(R, np.eye(3), atol=tolerance)

        # 2. Check if the translation vector is the zero vector
        is_t_zero = np.allclose(t, np.zeros(3), atol=tolerance)

        # 3. Check if the spinor matrix is the 2x2 identity matrix
        is_spinor_identity = np.allclose(spinor_mat_representation[idx], np.eye(2), atol=tolerance)

        # 4. Check if delta is 1 (no time reversal) using float comparison
        is_delta_one = np.isclose(delta_vec[idx], 1.0, atol=tolerance)

        # If all conditions are satisfied, this is the identity operation
        if is_R_identity and is_t_zero and is_spinor_identity and is_delta_one:
            identity_idx = idx
            break

    if identity_idx is None:
        raise ValueError("Identity operation not found in the provided magnetic space group operations.")

    return identity_idx

tol=1e-3
unit_cell_atoms=generate_atoms_in_unit_cell(parsed_config, magnetic_space_group_cart_spatial, lattice_basis,origin_cart, repr_s, repr_p, repr_d, repr_f, spinor_mat_representation,delta_vec,tol)

radius=parsed_config["truncation_radius"] # Cutoff distance in Cartesian coordinates
                         # Only atoms within this distance from center are considered neighbors
                         # FIXME: search_range must be sufficiently large to include all atoms within radius
                         # TODO: may need an algorithm to deal with this in the next version of code
                         # TODO: especially the vacuum layer length


search_range=10 # Number of unit cells to search in each direction
               # in general, for 1d, 2d, 3d problems, the search ranges is always [-10,10] × [-10,10] × [-10,10] ,
               #because there is vacuum  layer
               # for a 2d problem, the resulting range is often  [-10,10] × [-10,10] × [-1,1] ,
               #since the 2d layer has thickness
               # if atoms are found beyond vacuum layer, then set these hoppings=0
               # Larger values find more distant neighbors but increase computation time
               #TODO: should be computed from a0,a1, radius, vacuum layer length, etc


# ==============================================================================
# Find all neighbors for each atom in the unit cell
# ==============================================================================
# For each atom in the reference unit cell [0,0,0], find all neighboring atoms within
# the specified radius by searching through neighboring unit cells.
# This creates the hopping connectivity network for tight-binding calculations.

all_neighbors = {}  # Dictionary mapping unit cell atom index → list of neighbor atomIndex objects
                    # Key: integer index of center atom in unit_cell_atoms\
                    # Value: list of atomIndex objects representing all neighbors within radius
                    # Neighbors can be in different unit cells (n0, n1, n2)

for i, unit_atom in enumerate(unit_cell_atoms):
    # Find all neighbors within the specified radius for this center atom
    # 1. Searches through neighboring unit cells within search_range
    # 2. Constructs atomIndex objects for atoms in those cells
    # 3. Filters by distance (keeps only atoms within radius)
    # 4. Returns sorted list by distance
    neighbors = compute_dist(
        center_atom=unit_atom,  # Center atom (in unit cell [0,0,0])
        unit_cell_atoms=unit_cell_atoms,  # Template atoms to replicate in neighboring cells
        radius=radius,# Distance cutoff in Cartesian coordinates,
        search_range=search_range ,  # How many cells to search in each direction
    )
    # Store the neighbor list using the unit cell atom index as key
    # This creates a complete connectivity map: center_atom_idx --- [neighbor1, neighbor2, ...]
    all_neighbors[i] = neighbors
    # Print summary for each center atom
    print(f"Unit cell atom {i} ({unit_atom.wyckoff_instance_id}): found {len(neighbors)} neighbors within radius {radius}")


# ==============================================================================
# Find identity operation
# ==============================================================================
# Locate the identity operation E in the list of space group operations.
# The identity operation E = {identity matrix|0|delta=1} is characterized by:
# - Rotation part: 3×3 identity matrix (no operation)
# - Translation part: zero vector (no translation)
# - delta=1, no time reversal
#
# The identity operation index is crucial because:
# 1. It will be assigned to seed hoppings (root vertices in the constraint tree)
#    Seed hoppings are those containing identity operation
# 2. Root vertices in the vertex tree have hopping.operation_idx == identity_idx

#
# This index will be used throughout the code to:
# - Distinguish between seed hoppings and derived hoppings
# - Initialize root vertices in the constraint tree
# - Verify that orbital representations preserve identity (V[identity_idx] = identity matrix)


identity_idx = find_identity_operation(
    magnetic_space_group_cart_spatial,
    spinor_mat_representation,
    delta_vec,tolerance=1e-8
)
# ==============================================================================
# print atom orbital representations for all unit cell atoms
# ==============================================================================
print("\n" + "=" * 80)
print("PRINTING ATOM ORBITAL REPRESENTATIONS")
print("=" * 80)
for i, atom in enumerate(unit_cell_atoms):
    print(f"\nUnit cell atom {i} ({atom.wyckoff_instance_id}):")
    print(f"  {atom}")
    print(f"  Orbitals: {atom.get_orbital_names()}")
    if atom.orbital_representations:
        print(f"  Number of operations: {len(atom.orbital_representations)}")
        V_identity = atom.get_representation_matrix(identity_idx)
        print(f" Orbital representation's identity matrix shape: {V_identity.shape}")
        print(f" Orbital representation's identity present: {np.allclose(V_identity, np.eye(V_identity.shape[0]))}")

print("\n" + "=" * 80)
print("ORBITAL REPRESENTATION VERIFICATION COMPLETE")
print("=" * 80)

def print_tree(root, prefix="", is_last=True, show_details=True, max_depth=None, current_depth=0):
    """
    Print a constraint tree structure in a visual hierarchical format.

    Args:
        root: vertex object (root of tree or subtree)
        prefix: String prefix for indentation (used in recursion)
        is_last: Boolean indicating if this is the last child (affects connector style)
        show_details: Whether to show detailed hopping information (default: True)
        max_depth: Maximum depth to print (None = unlimited, default: None)
        current_depth: Current depth in recursion (internal use, default: 0)

    Tree Structure Symbols:
        ╔═══ ROOT     (root node)
        ├── CHILD    (middle child)
        └── CHILD    (last child)
        │           (vertical line for continuation)

    Example Output:
        ╔═══ ROOT: N[0,0,0] ← N[0,0,0], op=0, d=0.0000
        ├── CHILD (linear): N[0,0,0] ← N[1,0,0], op=1, d=2.5000
        ├── CHILD (linear): N[0,0,0] ← N[-1,1,0], op=2, d=2.5000
        └── CHILD (linear): N[0,0,0] ← N[0,-1,0], op=3, d=2.5000
    """
    # Check max depth
    if max_depth is not None and current_depth > max_depth:
        return

    # Determine node styling
    if root.is_root:
        node_label = "ROOT"
        connector = "╔═══ "
        detail_prefix = prefix
    else:
        node_label = f"CHILD ({root.type})"
        connector = "└── " if is_last else "├── "
        detail_prefix = prefix + ("    " if is_last else "│   ")

    # Build node description
    hop = root.hopping

    # Basic info: atom types and operation
    to_cell = f"[{hop.to_atom.n0},{hop.to_atom.n1},{hop.to_atom.n2}]"
    from_cell = f"[{hop.from_atom.n0},{hop.from_atom.n1},{hop.from_atom.n2}]"
    basic_info = f"{hop.to_atom.wyckoff_instance_id}{to_cell} ← {hop.from_atom.wyckoff_instance_id}{from_cell}"

    # Print main node line
    if show_details:
        print(f"{prefix}{connector}{node_label}: {basic_info}, "
              f"op={hop.operation_idx}, dist={hop.distance:.4f}")
    else:
        print(f"{prefix}{connector}{node_label}: op={hop.operation_idx}")

    # Print additional details if requested and this is root
    if show_details and root.is_root and current_depth == 0:
        print(f"{detail_prefix}    ├─ Type: {root.type}")
        print(f"{detail_prefix}    ├─ Children: {len(root.children)}")
        print(f"{detail_prefix}    └─ Distance: {hop.distance:.6f}")

    # Recursively print children
    if root.children:
        for i, child in enumerate(root.children):
            is_last_child = (i == len(root.children) - 1)

            # Determine new prefix for children
            if root.is_root:
                new_prefix = ""
            else:
                new_prefix = prefix + ("    " if is_last else "│   ")

            print_tree(child, new_prefix, is_last_child, show_details, max_depth, current_depth + 1)




def print_all_trees(roots_list, show_details=True, max_trees=None, max_depth=None):
    """
    Print all constraint trees in a formatted way.

    Args:
        roots_list: List of root vertex objects
        show_details: Whether to show detailed information (default: True)
        max_trees: Maximum number of trees to print (None = all, default: None)
        max_depth: Maximum depth to print for each tree (None = unlimited, default: None)
    """
    print("\n" + "=" * 80)
    print("CONSTRAINT TREE STRUCTURES")
    print("=" * 80)

    # CRITICAL FIX: Filter to only include actual roots (is_root == True)
    # ================================================================
    # ADD THIS LINE RIGHT HERE - it filters out grafted vertices
    actual_roots = [root for root in roots_list if root.is_root]

    # Print diagnostic if non-root vertices found in the list
    if len(actual_roots) < len(roots_list):
        print(f"\nNote: Input list contained {len(roots_list)} vertices")
        print(f"      Filtered to {len(actual_roots)} actual roots")
        print(f"      ({len(roots_list) - len(actual_roots)} vertices were grafted as hermitian children)\n")

    # Use actual_roots instead of roots_list for counting
    num_trees = len(actual_roots) if max_trees is None else min(max_trees, len(actual_roots))

    for i in range(num_trees):
        root = actual_roots[i]  # Changed from roots_list[i] to actual_roots[i]
        hop = root.hopping

        print(f"\n{'─' * 80}")
        print(f"Tree {i}: Distance = {hop.distance:.6f}, "
              f"Hopping: {hop.to_atom.position_name} ← {hop.from_atom.position_name}")
        print(f"{'─' * 80}")

        print_tree(root, show_details=show_details, max_depth=max_depth)

    if max_trees is not None and len(actual_roots) > max_trees:
        print(f"\n... and {len(actual_roots) - max_trees} more trees")

    print("\n" + "=" * 80)



# ==============================================================================
# Helper function for symmetry operations
# ==============================================================================

def cif_plus_translation(R,t,lattice_basis,n_vec,atom_cart):
    """
    Apply space group operation with lattice translation to an atom position.
    Computes the full symmetry transformation:
        r' = R @ r + t + n₀·a₀ + n₁·a₁ + n₂·a₂

    where:
        - R @ r is the rotation of the atom position
        - t is the fractional translation (origin shift) from the cif space group operation
        - n₀·a₀ + n₁·a₁ + n₂·a₂ is a lattice vector translation

    This is the complete symmetry operation that includes the additional lattice
    translation
    Args:
        R:  3×3 rotation matrix (in Cartesian coordinates, [0,0,0] origin)
        t:  3D translation vector (in Cartesian coordinates, [0,0,0] origin)
        lattice_basis:  3×3 array of primitive lattice basis vectors (each row is a basis vector)
                          expressed in Cartesian coordinates using [0,0,0] origin
        n_vec: Array [n₀, n₁, n₂] containing integer coefficients for lattice translation
        atom_cart: 3D Cartesian position of the atom (using [0,0,0] origin)

    Returns:
            transformed_cart, 3D Cartesian position after applying the full symmetry operation
    """
    # Extract the three primitive lattice basis vectors (each row is one basis vector)
    a0 = lattice_basis[0]  # First primitive basis vector
    a1 = lattice_basis[1]  # Second primitive basis vector
    a2 = lattice_basis[2]  # Third primitive basis vector
    # Extract the integer coefficients for the lattice translation
    # These determine how many unit cells to shift along each basis direction
    n0 = n_vec[0]
    n1 = n_vec[1]
    n2 = n_vec[2]
    # Apply the complete symmetry transformation:
    # 1. R @ atom_cart: Apply rotation to the atom position
    # 2. + t: Add the cif translation from the space group operation
    # 3. + n0*a0 + n1*a1 + n2*a2: Add the lattice vector translation
    #    This is the additional shift needed to preserve center atom invariance
    transformed_cart = R @ atom_cart + t + n0 * a0 + n1 * a1 + n2 * a2
    return transformed_cart

def is_lattice_vector(vector, lattice_basis, tolerance=1e-3):
    """
    Check if a vector can be expressed as an integer linear combination of lattice basis vectors.
     A vector v is a lattice vector if:
        v = n0*a0 + n1*a1 + n2*a2
    where n0, n1, n2 are integers and a0, a1, a2 are primitive lattice basis vectors.
    Args:
        vector:  3D vector to check (Cartesian coordinates)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using cif origin
        tolerance: Numerical tolerance for checking if coefficients are integers (default: 1e-3)

    Returns:
        tuple: (is_lattice, n_vector)
            - is_lattice (bool): True if vector is a lattice vector
            - n_vector (ndarray): The integer coefficients [n0, n1, n2]

    """
    # Extract basis vectors (each row is a basis vector)
    a0, a1, a2 = lattice_basis
    # Create matrix with basis vectors as columns
    lattice_matrix = np.column_stack([a0, a1, a2])
    # Solve: vector = lattice_matrix @ [n0, n1, n2]
    # So: [n0, n1, n2] = lattice_matrix^(-1) @ vector
    n_vector_float = np.linalg.solve(lattice_matrix, vector)
    # Round to nearest integers
    n_vector = np.round(n_vector_float)
    # Check if coefficients are integers (within tolerance)
    is_lattice = np.allclose(n_vector_float, n_vector, atol=tolerance)
    return is_lattice, n_vector

def check_center_invariant(center_atom, operation_idx, magnetic_space_group_cart_spatial,
                           lattice_basis, tolerance=1e-3):
    """
    Check if a center atom is invariant under a specific magnetic space group spatial part operation.
     An atom is invariant if the symmetry operation maps it to itself, possibly
     translated by a lattice vector. The actual operation is:
        r' = R @ r + t + n0*a0 + n1*a1 + n2*a2
    where n0, n1, n2 are integers and a0, a1, a2 are primitive lattice basis vectors.

    For invariance, we need: r' = r, which means:
        R @ r + t + n0*a0 + n1*a1 + n2*a2 = r
        => (R - I) @ r + t = -(n0*a0 + n1*a1 + n2*a2)

    Args:
        center_atom:  atomIndex object representing the center atom
        operation_idx: Index of the magnetic space group operation to check
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
                                 using cif origin (shape: num_ops × 3 × 4)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using cif origin
        tolerance: Numerical tolerance for comparison (default: 1e-3)


    Returns:
        tuple: (is_invariant, n_vector)
            - is_invariant (bool): True if the atom is invariant under the operation
            - n_vector (ndarray): The integer coefficients [n0, n1, n2] for lattice translation
    """
    # Extract the rotation matrix R and translation vector t from the space group operation
    R, t =get_rotation_translation(magnetic_space_group_cart_spatial, operation_idx)
    # Get center atom's Cartesian position (using cif origin)
    r_center = center_atom.cart_coord
    # Compute the left-hand side of the invariance equation:
    # (R - I) @ r + t
    # For invariance, this must equal -(n0*a0 + n1*a1 + n2*a2) for integer n0, n1, n2
    lhs = (R - np.eye(3)) @ r_center + t
    # Check if -lhs can be expressed as an integer linear combination of lattice basis vectors
    # If yes, then there exists a lattice translation that makes the atom invariant
    # n_vector contains the integer coefficients [n0, n1, n2]
    is_invariant, n_vector = is_lattice_vector(-lhs, lattice_basis, tolerance)
    return is_invariant, n_vector

def get_next_for_center(center_atom, seed_atom, center_seed_distance, magnetic_space_group_cart_spatial,
                        operation_idx, parsed_config, tolerance=1e-3):
    """
    Apply a magnetic space group spatial part operation to a seed atom, conditioned on center atom invariance.
    This function implements a three-step validation process:
    1. Check if the center atom is invariant under the magnetic space group  spatial operation
       (usually with lattice translation). This determines the lattice shift n_vec.
    2. If invariant, apply the SAME operation (with the SAME n_vec) to the seed atom
        to generate an atom's Cartesian coordinate. This atom should be symmetry-equivalent
        to the seed atom.
    3. Verify that the transformed seed maintains the same distance from center.

    Physical Context:
    Given a seed hopping (center ← seed), this function applies a symmetry operation
    to generate a potentially equivalent hopping (center ← transformed_seed). The
    transformed position is only returned if it preserves the hopping distance,
    confirming it belongs to the same equivalence class, if the atom type also matches
    Args:
        center_atom: atomIndex object for the center atom (target of the hopping)
        seed_atom: atomIndex object for the seed neighbor atom (origin of seed hopping)
        center_seed_distance:  Pre-computed distance from center to seed atom
                               (avoids redundant computation across operations)
        magnetic_space_group_cart_spatial:  List of magnetic space group spatial part matrices in Cartesian coordinates
                                using cif origin (shape: num_ops × 3 × 4)
        operation_idx: Index of the magnetic space group operation to apply
        parsed_config: Configuration dictionary containing lattice_basis
        tolerance: Numerical tolerance for invariance and distance checks (default: 1e-3)
        verbose: Whether to print debug information (default: False)

    Returns:
        numpy.ndarray, numpy.ndarray  or None:
            - Transformed Cartesian coordinates if:
              (a) center is invariant under this operation, AND
              (b) transformation preserves the center-seed distance
            - None otherwise (operation doesn't generate a valid equivalent hopping)
    """
    # ==============================================================================
    # Extract space magnetic group operation components
    # ==============================================================================
    # Get the rotation matrix R and translation vector b from the magnetic space group operation
    # The operation is represented as [R|b] in Cartesian coordinates
    R, b = get_rotation_translation(magnetic_space_group_cart_spatial, operation_idx)

    # ==============================================================================
    # Get lattice basis vectors
    # ==============================================================================
    # Extract the primitive lattice basis vectors from configuration
    # These are the fundamental translation vectors a0, a1, a2 of the crystal
    lattice_basis = np.array(parsed_config['lattice_basis'])
    # ==============================================================================
    # STEP 1: Check center atom invariance
    # ==============================================================================
    # Determine if the center atom is invariant under this space group operation,
    # usually with an additional lattice translation.
    ## Mathematical condition for invariance:
    #   R @ r_center + b + n_vec · [a0, a1, a2] = r_center
    # where n_vec = [n0, n1, n2] are integer coefficients for lattice translation
    #
    # This ensures that the symmetry operation preserves the hopping target -
    # the center atom remains at the same atomic site.
    #
    # Returns:
    #   is_invariant: True if center atom is invariant (with some lattice shift)
    #   n_vec: The required lattice translation [n0, n1, n2] for invariance
    is_invariant, n_vec = check_center_invariant(
        center_atom,
        operation_idx,
        magnetic_space_group_cart_spatial,
        lattice_basis,
        tolerance
    )
    # ==============================================================================
    # STEP 2: Apply operation to seed atom (only if center is invariant)
    # ==============================================================================
    if is_invariant:
        # Center atom is invariant under this operation (with lattice shift n_vec)
        # Now apply the SAME complete transformation to the seed atom:
        #   r_transformed = R @ r_seed + b + n_vec · [a0, a1, a2]
        #
        # KEY INSIGHT: We must use the SAME n_vec for the seed!
        # This ensures the hopping vector (r_center - r_seed) transforms consistently,
        # preserving the relative geometry of the hopping.
        ## The transformed position may correspond to an atom that is symmetry-equivalent
        # to the seed atom (same species, equivalent local environment).
        seed_cart_coord = seed_atom.cart_coord  # Original seed position
        # Apply the full symmetry transformation: R @ r + b + lattice_shift
        # This generates a Cartesian coordinate that may represent a symmetry-equivalent atom
        next_cart_coord = cif_plus_translation(R, b, lattice_basis, n_vec, seed_cart_coord)
        # ==============================================================================
        # STEP 3: Verify the transformation preserves hopping distance (isometry check)
        # ==============================================================================
        # Calculate the distance from the transformed position to center
        # For a valid symmetry operation (isometry), this must equal the original distance
        new_center_seed_dist = np.linalg.norm(next_cart_coord - center_atom.cart_coord, ord=2)
        # Check if the hopping distance is preserved
        # This verifies that: |r_center - (R @ r_seed + b + lattice_shift)| = |r_center - r_seed|
        dist_is_equal = (np.abs(new_center_seed_dist - center_seed_distance) < tolerance)
        # Only return the transformed position if distance is preserved
        # This ensures the generated coordinate represents a symmetry-equivalent atom
        if dist_is_equal == True:
            return (deepcopy(next_cart_coord), deepcopy(n_vec))
        else:
            # Distance not preserved - this indicates a numerical error or
            # inconsistency in the symmetry operation (shouldn't happen in practice)
            return None,None

    else:
        # ==============================================================================
        # Center atom is NOT invariant - operation invalid for this hopping
        # ==============================================================================
        # The center atom does not map to itself (even with lattice translations)
        # under this magnetic space group operation. Therefore, abandon this symmetry operation
        return None,None


def search_one_equivalent_atom(seed_atom,target_cart_coord, neighbor_atoms_copy, tolerance=1e-3):
    """
    Search for an atom in the neighbor_atoms_copy set whose Cartesian coordinate matches the target.
    This function is used to find which actual neighbor atom corresponds to a transformed
    position generated by a symmetry operation. If a match is found, it confirms that
    the symmetry operation maps the seed atom to an existing neighbor atom.
    Args:
        seed_atom: atomIndex object for the seed neighbor atom (origin of seed hopping)
        target_cart_coord: 3D Cartesian coordinate to search for (numpy array)
                            This is  the result of applying a symmetry operation
                            to a seed atom's position
        neighbor_atoms_copy: set of atomIndex objects representing all neighbors
                         of a center atom within some cutoff radius
        tolerance:   Numerical tolerance for coordinate comparison (default: 1e-3)
                    Two positions are considered identical if their Euclidean distance
                    is less than this tolerance


    Returns:
        atomIndex or None:
          - The matching neighbor atom if found (coordinate matches within tolerance)
            IMPORTANT: Returns a REFERENCE (not a copy) to the atomIndex object in neighbor_atoms
          - None if no match is found (transformed position doesn't correspond to
            any actual neighbor atom)

    """
    # Iterate through all neighbor atoms in the set
    for neighbor in neighbor_atoms_copy:
        # Compute Euclidean distance between target position and this neighbor's position
        distance = np.linalg.norm(target_cart_coord - neighbor.cart_coord, ord=2)
        # Check if the distance is within tolerance (positions match)
        if distance < tolerance and seed_atom.position_name == neighbor.position_name:
            # Return a REFERENCE (pointer in C sense, reference in C++ sense) to the matching neighbor atom
            # This is NOT a deep copy - it's the same object that exists in neighbor_atoms_copy
            # This allows the caller to use this reference to remove the atom from neighbor_atoms_copy
            return neighbor
    return None


def get_equivalent_sets_for_one_center_atom(center_atom_idx, unit_cell_atoms, all_neighbors,
                                                magnetic_space_group_cart_spatial, identity_idx,
                                                tolerance=1e-3):
    """
    Partition all neighbors of 1 center atom into equivalence classes based on symmetry.
    Each equivalence class contains center atom's neighbors related by magnetic space group spatial part operations.
    Algorithm:
    ---------
    1. Pop a seed atom from the remaining neighbors (arbitrary choice)
    2. Apply all magnetic space group spatial part operations to find symmetry-equivalent neighbors
    3. Group these equivalent neighbors together into one equivalence class
    4. Repeat until all neighbors are classified

    CRITICAL: Reference Handling
    ============================
    This function works with REFERENCES to atomIndex objects throughout:
    - neighbor_atoms_copy is a set of references to DEEP-COPIED atomIndex objects
    - seed_atom = set.pop() returns a reference to one of these copied objects
    - matched_neighbor from search is also a reference to one of these copied objects
    - equivalence_classes stores tuples containing references to these copied objects

    Why deep copy all_neighbors[center_atom_idx]?
    ---------------------------------------------
    We deep copy to DECOUPLE from the input:
    1. The input all_neighbors should remain unchanged (it may be used elsewhere)
    2. We destructively remove atoms from neighbor_atoms_copy as we classify them
    3. Deep copy creates NEW atomIndex objects (independent of the input)
    4. After deep copy:
        - all_neighbors[center_atom_idx] still has all its original atomIndex objects
        - neighbor_atoms_copy has completely separate atomIndex objects with same data
        - Modifying neighbor_atoms_copy has NO effect on all_neighbors

    Args:
        center_atom_idx: Index of the center atom in unit_cell_atoms
        unit_cell_atoms: List of all atomIndex objects in the unit cell
        all_neighbors: Dictionary mapping center atom index → list of neighbor atomIndex objects
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
        identity_idx: Index of the identity operation
        tolerance: Numerical tolerance for comparisons (default: 1e-3)


    Returns:
        List of equivalence classes, where each class is a list of tuples:
        (matched_neighbor, operation_idx, n_vec)
        where:
            - matched_neighbor: REFERENCE to deep-copied atomIndex object
            - operation_idx: Space group operation that maps seed → matched_neighbor
            - n_vec: Lattice translation vector [n₀, n₁, n₂] for this transformation

    """
    # ==============================================================================
    # Initialize working variables
    # ==============================================================================
    # Extract reference to center atom from unit cell
    # This is a REFERENCE (not copied) - center_atom points to the same object in unit_cell_atoms
    center_atom = unit_cell_atoms[center_atom_idx]
    # print(f"center_atom={center_atom}")
    # Create a working copy of neighbors as a set
    # IMPORTANT: Deep copy to DECOUPLE from input all_neighbors
    # ----------------------------------------------------------
    # Why deep copy?
    # - We will destructively remove atoms from neighbor_atoms_copy as we classify them
    # - We must NOT modify the input all_neighbors (caller may need it unchanged)
    # - Deep copy creates entirely NEW atomIndex objects (different memory addresses)
    #   with the same data as the originals
    #
    # Memory structure after deep copy:
    # - all_neighbors[center_atom_idx] = [obj_A, obj_B, obj_C, ...]  (original objects)
    # - neighbor_atoms_copy = {obj_A', obj_B', obj_C', ...}  (NEW copied objects)
    # - obj_A and obj_A' are DIFFERENT objects at DIFFERENT memory addresses
    # - obj_A and obj_A' have the SAME data (same coordinates, same element, etc.)
    # - Removing obj_A' from neighbor_atoms_copy does NOT affect all_neighbors
    #
    # Why set instead of list?
    # - O(1) removal with set.remove() vs O(n) with list.remove()
    # - No duplicates guaranteed
    # - Order doesn't matter (symmetry operations find all equivalents)
    neighbor_atoms_copy = set(deepcopy(all_neighbors[center_atom_idx]))
    # Store all equivalence classes (list of lists of tuples)
    equivalence_classes = []
    # Class ID counter (increments for each new equivalence class found)
    class_id = 0
    # ==============================================================================
    # Main loop: Partition neighbors into equivalence classes
    # ==============================================================================
    # Continue until all neighbors are classified into equivalence classes
    # Each iteration creates one equivalence class and removes its members from neighbor_atoms_copy
    while len(neighbor_atoms_copy) != 0:
        # ==============================================================================
        # STEP 1: Select seed atom for this equivalence class
        # ==============================================================================
        # Pop one seed atom from neighbor_atoms_copy
        # This will be the representative atom for this equivalence class
        #
        # CRITICAL: set.pop() returns a REFERENCE, not a copy
        # ------------------------------------------------
        # - set.pop() removes and returns a reference to an arbitrary element
        # - Order is implementation-dependent (hash table internals, not guaranteed)
        # - Returns a REFERENCE to one of the deep-copied atomIndex objects
        # - The atomIndex object is removed from the set but still exists in memory
        # - seed_atom now holds a reference to that object
        #
        # Example:
        # -------
        # Before: neighbor_atoms_copy = {obj_A', obj_B', obj_C'}
        # After:  seed_atom = obj_A' (reference to the copied object)
        #         neighbor_atoms_copy = {obj_B', obj_C'}
        #
        # Remember: obj_A' is a COPY (independent of the original obj_A in all_neighbors)
        #
        # The specific choice doesn't matter - symmetry operations will find all equivalent neighbors
        seed_atom = neighbor_atoms_copy.pop()
        # Pre-compute the distance from center to seed (used for all operations)
        # This distance must be preserved by symmetry operations (isometry)
        center_seed_distance = np.linalg.norm(center_atom.cart_coord - seed_atom.cart_coord, ord=2)
        # ==============================================================================
        # Initialize the current equivalence class
        # ==============================================================================
        # List of tuples: (neighbor_atom_reference, operation_idx, n_vec)
        current_equivalence_class = []
        # FIX: Add the seed atom itself to the equivalence class!
        # It corresponds to the identity operation and zero lattice shift.
        current_equivalence_class.append((seed_atom, identity_idx, np.array([0, 0, 0])))
        # ==============================================================================
        # STEP 2: Find all symmetry-equivalent neighbors
        # ==============================================================================
        # Iterate through all space group operations to find atoms equivalent to seed
        # Skip the identity operation since we already added the seed atom
        for operation_idx in range(len(magnetic_space_group_cart_spatial)):
            # Skip identity operation (already handled)
            if operation_idx == identity_idx:
                continue
            # Apply the space group operation to the seed atom
            # This generates a transformed position that may correspond to another neighbor
            # Returns (transformed_coord, n_vec) if valid, None otherwise
            result = get_next_for_center(
                center_atom=center_atom,
                seed_atom=seed_atom,
                center_seed_distance=center_seed_distance,
                magnetic_space_group_cart_spatial=magnetic_space_group_cart_spatial,
                operation_idx=operation_idx,
                parsed_config=parsed_config,
                tolerance=tolerance
            )
            # ==============================================================================
            # Process valid transformation results
            # ==============================================================================
            # If transformation is valid (center invariant, distance preserved)
            if result[0]  is not None:
                # Unpack the transformed coordinate and lattice shift vector
                # transformed_coord: 3D Cartesian position after applying symmetry operation
                # n_vec: Lattice translation [n₀, n₁, n₂] needed to preserve center invariance
                transformed_coord, n_vec = result
                # ==============================================================================
                # Search for matching neighbor in the remaining unclassified set
                # ==============================================================================
                # Search for this transformed position among the remaining neighbors
                # CRITICAL: matched_neighbor is a REFERENCE, not a copy
                # ---------------------------------------------------
                # search_one_equivalent_atom() returns:
                # - A REFERENCE to an atomIndex object in neighbor_atoms_copy if match found
                # - None if no match found
                #
                # This reference is ESSENTIAL for set.remove() to work:
                # - Python's set.remove() uses object identity (memory address)
                # - We need the EXACT SAME object reference that's in the set
                # - A copy wouldn't work (different object, different identity)
                #
                # Remember: matched_neighbor references a COPIED atomIndex object (obj_X')
                # NOT an original from all_neighbors (obj_X)
                matched_neighbor = search_one_equivalent_atom(
                    seed_atom,
                    target_cart_coord=transformed_coord,
                    neighbor_atoms_copy=neighbor_atoms_copy,
                    tolerance=tolerance
                )
                # ==============================================================================
                # Add matched neighbor to equivalence class
                # ==============================================================================
                # If we found a matching neighbor in the remaining set
                if matched_neighbor is not None:
                    # Add to current equivalence class
                    # Store tuple: (reference to matched_neighbor, operation_idx, copy of n_vec)
                    # - matched_neighbor: REFERENCE to a deep-copied atomIndex object (from neighbor_atoms_copy)
                    # - operation_idx: Which space group operation maps seed → matched_neighbor
                    # - deepcopy(n_vec): Copy of lattice translation vector (n_vec is numpy array, mutable)
                    current_equivalence_class.append((matched_neighbor, operation_idx, deepcopy(n_vec)))
                    # Remove from the working set (it's now classified)
                    # CRITICAL: This only works because matched_neighbor is a REFERENCE
                    # ----------------------------------------------------------------
                    # set.remove() searches for object by identity (memory address)
                    # - matched_neighbor points to the exact same object in neighbor_atoms_copy
                    # - Python finds the object by comparing memory addresses (fast, O(1))
                    # - If matched_neighbor were a copy, remove() would raise KeyError
                    #
                    # After removal:
                    # - The atomIndex object still exists in memory (referenced by matched_neighbor
                    #   and by the tuple in current_equivalence_class)
                    # - It's just no longer in the neighbor_atoms_copy set
                    # - The original object in all_neighbors is completely unaffected
                    neighbor_atoms_copy.remove(matched_neighbor)
        # ==============================================================================
        # Complete this equivalence class
        # ==============================================================================
        # Add the completed equivalence class to the list
        # equivalence_classes is a list of lists of tuples
        # Each tuple contains: (reference to deep-copied atomIndex, operation_idx, n_vec)
        equivalence_classes.append(current_equivalence_class)
        # Increment class ID for next equivalence class
        class_id += 1

    # ==============================================================================
    # Return results
    # ==============================================================================
    return equivalence_classes


def equivalent_class_to_hoppings(one_equivalent_class, center_atom,
                                  magnetic_space_group_cart_spatial,spinor_mat_representation,delta_vec, identity_idx):
    """
    Convert an equivalence class of neighbor atoms into hopping objects.
    Each neighbor atom in the equivalence class is saved into a hopping object.
    The hopping contains all symmetry information (operation index, rotation, translation, lattice shift).

    This function transforms the raw equivalence class data (tuples of neighbor atoms,
    operations, and lattice shifts) into structured hopping objects that encapsulate
    all information needed for one class of equivalent hoppings (center ← neighbor) and symmetry constraints.

    Args:
        one_equivalent_class: List of tuples (neighbor_atom, operation_idx, n_vec)
                              where:
                                - neighbor_atom: atomIndex object for the neighbor
                                - operation_idx: Index of space group operation that maps
                                              seed atom to this neighbor
                                - n_vec: Array [n₀, n₁, n₂] of lattice translation coefficients
        center_atom: atomIndex object for the center atom (hopping destination)
                    All hoppings in this equivalence class have the same center atom
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
                                using cif origin (shape: num_ops × 3 × 4)
                                Used to extract rotation R and translation t for each operation
        spinor_mat_representation: spinor part matrices
        delta_vec: indicating time reversal, values are ±1
        identity_idx: Index of the identity operation in magnetic_space_group_cart_spatial
                     Used to identify which hopping is the seed (root of constraint tree)

    Returns:
        List of hopping objects (deep copied for complete independence).
        Each hopping represents: center ← neighbor
        The list contains:
            - One seed hopping (with operation_idx == identity_idx, is_seed=True)
            - Multiple derived hoppings (with other operation indices, is_seed=False)
        All hoppings in the list have the same distance (up to numerical precision)
    Deep Copy Strategy:
        This function returns a DEEP COPY of the entire hopping list to ensure
        complete independence between the returned data and any internal state.
        Two-level protection:
            1. Each hopping object is deep copied before adding to the list
            2. The entire list is deep copied before returning
        This guarantees:
            - No shared references to the list container
            - No shared references to hopping objects
            - No shared references to atom objects or numpy arrays
            - Caller has complete ownership and can modify freely


    """
    # Initialize hopping list
    hoppings = []
    # Convert each equivalence class member to a hopping object
    for neighbor_atom, operation_idx, n_vec in one_equivalent_class:
        # Extract rotation matrix R and translation vector t for this operation
        # The magnetic space group spatial part operation [R|t] transforms the seed neighbor to this neighbor
        # R, t = get_rotation_translation(magnetic_space_group_cart_spatial, operation_idx)
        # spinor_mat=spinor_mat_representation[operation_idx]
        # delta=delta_vec[operation_idx]
        # Determine if this is the seed hopping (generated by identity operation)
        # The seed hopping serves as the root of the constraint tree
        is_seed = (operation_idx == identity_idx)
        # Create hopping object: center ← neighbor
        # This represents the tight-binding hopping from neighbor to center atom
        hop = hopping(
            to_atom=deepcopy(center_atom),  # Destination: center atom (deep copied)
            from_atom=deepcopy(neighbor_atom),  # Source: neighbor atom (deep copied),
            operation_idx=operation_idx,  # Space group operation index (immutable int),
            # rotation_matrix=deepcopy(R),  # 3×3 rotation matrix from cif (deep copied)
            # translation_vector=deepcopy(t),  # 3D translation vector from cif (deep copied)
            n_vec=deepcopy(n_vec),  # Additional lattice shift [n₀, n₁, n₂] (deep copied)
            # spinor_mat=spinor_mat,
            # delta=delta,
            is_seed=is_seed
        )
        # Compute the Euclidean distance from neighbor to center
        # All hoppings in this equivalence class should have the same distance
        hop.compute_distance()
        # Add this hopping to the list
        # Deep copy hopping before adding to list (first level of protection)
        hoppings.append(deepcopy(hop))

    # Deep copy entire list before returning (second level of protection)
    # This ensures complete independence: both list structure and contents are copied
    return deepcopy(hoppings)


def hopping_to_vertex(hopping,identity_idx,type_linear):
    """
    Convert a hopping object to a vertex object., for equivalent class step
    Args:
        hopping:  hopping object to convert
        identity_idx:  Index of the identity operation
        type_linear:  string "linear"

    Returns:
        vertex object (deep copied for independence)


    """
    # Determine constraint type based on whether this is a seed hopping
    if hopping.is_seed == True:
        constraint_type = None  # Root vertex has no parent constraint
    else:
        constraint_type = type_linear  # Derived from symmetry operation
    # Create vertex with no parent (parent will be set when building tree)
    new_vertex = vertex(hopping,
                        constraint_type,
                        identity_idx,
                        parent=None)

    return deepcopy(new_vertex)


def one_equivalent_hopping_class_to_root(one_equivalent_hopping_class, identity_idx, type_linear):
    """
    Convert an equivalent hopping class into a constraint tree.

    This function:
    1. Converts all hoppings to vertex objects
    2. Finds the root vertex (seed hopping with identity operation)
    3. Connects all derived vertices as linear children of the root
    4. Returns the root vertex (which contains references to all children)

    Tree Structure Created:
    ----------------------
                    Root (seed, identity operation)
                     |
         +-----------+-----------+-----------+
         |           |           |           |
      Child 0     Child 1     Child 2     Child 3
     (linear)    (linear)    (linear)    (linear)

    Each child is derived from root by a symmetry operation.
    Args:
        one_equivalent_hopping_class: List of hopping objects (all symmetry-equivalent)
        identity_idx: Index of the identity operation
        type_linear: String identifier for linear constraint type (string "linear")

    Returns:
        vertex object: Root of the constraint tree (contains references to all children)
    Raises:
        ValueError: If no root vertex found (no seed hopping in the class)
    """
    # ==============================================================================
    # STEP 1: Convert all hoppings to vertices
    # ==============================================================================
    vertex_list = [hopping_to_vertex(one_hopping, identity_idx, type_linear) for one_hopping in
                   one_equivalent_hopping_class]
    # ==============================================================================
    # STEP 2: Find the root vertex (seed hopping)
    # ==============================================================================
    tree_root = None
    derived_vertices = []  # List to store non-root vertices
    for one_vertex in vertex_list:
        if one_vertex.is_root == True:
            if tree_root is not None:
                # Multiple roots found - this shouldn't happen
                raise ValueError("Multiple root vertices found in equivalence class! "
                                 "Each class should have exactly one seed hopping.")
            tree_root = one_vertex
        else:
            derived_vertices.append(one_vertex)

    # ==============================================================================
    # STEP 3: Validate that root was found
    # ==============================================================================
    if tree_root is None:
        raise ValueError("No root vertex found in equivalence class! "
                         f"Identity operation (idx={identity_idx}) not present.")
    # ==============================================================================
    # STEP 4: Connect all derived vertices as children of root
    # ==============================================================================
    # CRITICAL: Use add_child() to establish bidirectional parent-child relationships
    # This creates REFERENCES (not copies) between root and children
    for i, child_vertex in enumerate(derived_vertices):
        tree_root.add_child(child_vertex)

    # ==============================================================================
    # STEP 5: Return the root vertex
    # ==============================================================================
    # IMPORTANT: Return tree_root WITHOUT deep copying
    # ------------------------------------------------
    # The tree_root contains REFERENCES to its children via tree_root.children
    # Deep copying would break these parent-child references
    # Caller receives the actual root vertex object with intact tree structure
    return tree_root


def atom_equal(atom1, atom2,tolerence=1e-3):
    """
    check if two atoms occupy the same position
    Args:
        atom1:
        atom2:

    Returns:

    """
    dist_diff=np.linalg.norm(atom1.cart_coord-atom2.cart_coord,ord=2)
    if dist_diff<tolerence and atom1.position_name==atom2.position_name:
        return True
    else:
        return False


def apply_full_transformation_and_check_position(atom1,atom2,R,t,lattice_basis,n_vec,tolerance=1e-3):
    #checks if full transformation applied to atoms1 goes to atom2
    atom1_pos=atom1.cart_coord
    atom2_pos=atom2.cart_coord

    atom1_transformed_pos=cif_plus_translation(R,t,lattice_basis,n_vec,atom1_pos)
    diff=np.linalg.norm(atom1_transformed_pos-atom2_pos,ord=2)
    if diff<tolerance:
        return True
    else:
        return False


def check_hopping_linear(hopping1,hopping2, magnetic_space_group_cart_spatial,
                            lattice_basis, tolerance=1e-3):
    """
    Check if hopping2 is related to hopping1 by a space group symmetry operation.
     For tight-binding models, a linear symmetry constraint implies:
        (i) for delta=1, no time reversal,
            T(hopping2) = [V1(g)⊗U(g)] @ T(hopping1) @ [V2(g)†⊗U(g)†]
        (ii) for delta=-1, there is time reversal
            T(hopping2) = [V1(g)⊗~U(g)] @ T*(hopping1) @ [V2(g)†⊗~U(g)†]
    Geometrically, this function checks if the displacement vector of hopping2
    is the result of applying a magnetic space group spatial part operation plus a lattice shift to
    the displacement vector of hopping1.

    Mathematical Condition:
    ----------------------
    Given hopping1 vector: r1 = center1 - neighbor1
    Given hopping2 vector: r2 = center2 - neighbor2

    hopping2 is linearly related to hopping1 if there exists a space group
    operation g = (R|t) and lattice shift n_vec = [n0, n1, n2] such that:
        R @ r1 + t + n_vec·[a0,a1,a2] = r2
    EDGE CASE (Self-Hopping / On-Site):
    -----------------------------------
    If dist1 = dist2 = 0 (hopping is from an atom to itself), the hopping vector is 0.
    The condition simplifies to checking if the operation maps atom1 to atom2:
        R @ pos_atom1 + t + n_vec·basis = pos_atom2

    Args:
        hopping1: First hopping object (reference hopping)
        hopping2: Second hopping object (candidate symmetry equivalent)
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
                                           using cif origin (shape: num_ops × 3 × 4)

        lattice_basis: Primitive lattice basis vectors (3×3 array), each row is a basis vector
        tolerance: Numerical tolerance for comparison (default: 1e-3)

    Returns:
        tuple: (is_linear, operation_idx, n_vec, delta)
        - is_linear (bool): True if hopping2 is related to hopping1 via symmetry
        - operation_idx (int or None): Index of the magnetic space group operation
        - n_vec (ndarray or None): Lattice translation vector [n0, n1, n2]

    """
    # ==============================================================================
    # STEP 1: Extract atoms and validate types
    # ==============================================================================
    # hopping1: to_atom1 (center) ← from_atom1 (neighbor)
    to_atom1 = hopping1.to_atom
    from_atom1 = hopping1.from_atom

    # hopping2: to_atom2 (center) ← from_atom2 (neighbor)
    to_atom2 = hopping2.to_atom
    from_atom2 = hopping2.from_atom

    to_atom1_position_name = to_atom1.position_name
    from_atom1_position_name = from_atom1.position_name

    to_atom2_position_name = to_atom2.position_name
    from_atom2_position_name = from_atom2.position_name

    dist1 = hopping1.distance
    dist2 = hopping2.distance

    # Check 1: Hopping distances must be identical (isometry)
    if np.abs(dist1 - dist2) > tolerance:
        return False, None, None
    # Check 2: Atom wyckoff position must match for a valid symmetry operation
    if to_atom1_position_name != to_atom2_position_name or from_atom1_position_name != from_atom2_position_name:
        return False, None, None

    # ==============================================================================
    # STEP 2: Handle Edge Case (Self-Hopping / On-Site Terms)
    # ==============================================================================
    is_self_hopping = (dist1 < tolerance)
    if is_self_hopping:
        # For self-hopping, center and neighbor are the same atom
        # We need to check if the operation maps atom1 to atom2.
        # Equation: R @ pos_atom1 + t + n_vec·basis = pos_atom2
        pos_atom1 = to_atom1.cart_coord
        pos_atom2 = to_atom2.cart_coord
        for op_idx in range(len(magnetic_space_group_cart_spatial)):
            R, t = get_rotation_translation(magnetic_space_group_cart_spatial, op_idx)
            # Apply operation to atom1 position
            transformed_pos = R @ pos_atom1 + t
            # Calculate required shift to reach atom2
            required_lattice_shift = pos_atom2 - transformed_pos
            is_lattice, n_vec =is_lattice_vector(
                required_lattice_shift,
                lattice_basis,
                tolerance
            )
            if is_lattice:
                return True, op_idx, n_vec.astype(int)
        # If loop finishes for self-hopping without match
        return False, None, None

    # ==============================================================================
    # STEP 3: Standard Case (Inter-atomic Hopping)
    # ==============================================================================
    # Displacement vector for hopping1
    # print("entering here")
    for op_idx in range(len(magnetic_space_group_cart_spatial)):
        # Extract rotation R and translation t from space group operation
        R, t = get_rotation_translation(magnetic_space_group_cart_spatial, op_idx)
        # Apply rotation and translation to to_atom1.cart_coord\
        # transformed = R @ r1 + t
        transformed_to_atom1_pos = R @ to_atom1.cart_coord + t
        # Calculate required lattice shift
        # We need: transformed_vec + n_vec·basis = hopping_vec2
        # Therefore: n_vec·basis = hopping_vec2 - transformed_vec
        required_lattice_shift = to_atom2.cart_coord - transformed_to_atom1_pos
        # Check if required_lattice_shift is a lattice vector
        is_lattice, n_vec = is_lattice_vector(
            required_lattice_shift,
            lattice_basis,
            tolerance
        )
        if is_lattice:
            # check if this operation maps from_atom1 to from_atom2
            from_atoms_match = apply_full_transformation_and_check_position(from_atom1, from_atom2, R, t, lattice_basis,
                                                                            n_vec,
                                                                           tolerance)
            if from_atoms_match:
                return True, op_idx, n_vec.astype(int)
            else:
                continue

    # ==============================================================================
    # No linear relationship found
    # ==============================================================================
    return False, None, None




def check_hopping_hermitian(hopping1,hopping2, magnetic_space_group_cart_spatial,
                            lattice_basis, tolerance=1e-3):
    """
     Check if hopping2 is the Hermitian conjugate of hopping1.
    For tight-binding models, Hermiticity requires:
        H† = H  =>  T(i ← j) = T(j ← i)†

    This function checks if hopping2 corresponds to the reverse direction of hopping1
    under some magnetic space group spatial part operation with lattice translation.


    Mathematical Condition:
    ----------------------
    Given hopping1: center1 ← neighbor1
    And hopping2: center2 ← neighbor2
    hopping2 is Hermitian conjugate of hopping1 if there exists a magnetic space group
    spatial part  operation g = (R|t) and lattice shift n_vec = [n0, n1, n2] such that:
    1. The conjugate of hopping2 (neighbor2 ← center2) equals the transformed hopping1
    2. Specifically: R @ (center1 - neighbor1) + t + n_vec·[a0,a1,a2] = neighbor2 - center2

    This means the hopping vector transforms consistently under the symmetry operation.

    Args:
        hopping1: First hopping object (reference hopping)
        hopping2: Second hopping object (candidate Hermitian conjugate)
        magnetic_space_group_cart_spatial: List of magnetic space group spatial part matrices in Cartesian coordinates
                                           using cif origin (shape: num_ops × 3 × 4)

        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using cif origin
        tolerance: Numerical tolerance for comparison (default: 1e-3)

    Returns:
        tuple: (is_hermitian, operation_idx, n_vec, delta)
        - is_hermitian (bool): True if hopping2 is Hermitian conjugate of hopping1
        - operation_idx (int or None): Index of the space group operation that
                                        relates hopping1 to hopping2, or None if not Hermitian conjugate
        - n_vec (ndarray or None): Lattice translation vector [n0, n1, n2],
                                   or None if not Hermitian conjugate

    Example:
        For hBN with hopping1: N[0,0,0] ← B[0,0,0]
        and hopping2: B[0,0,0] ← N[0,0,0]
        These are Hermitian conjugates under identity operation with zero lattice shift.
    """
    # ==============================================================================
    # STEP 1: Get atoms from both hoppings
    # ==============================================================================
    # hopping1: to_atom1 (center) ← from_atom1 (neighbor)
    to_atom1 = hopping1.to_atom
    from_atom1 = hopping1.from_atom
    # hopping2: to_atom2 (center) ← from_atom2 (neighbor)
    # For Hermiticity, we need the CONJUGATE (reverse direction)
    # conjugate of hopping2: to_atom2c (becomes center) ← from_atom2c (becomes neighbor)
    to_atom2c, from_atom2c = hopping2.conjugate()

    to_atom1_position_name = to_atom1.position_name
    from_atom1_position_name = from_atom1.position_name

    to_atom2c_position_name = to_atom2c.position_name
    from_atom2c_position_name = from_atom2c.position_name

    dist1 = hopping1.distance
    dist2 = hopping2.distance
    if np.abs(dist1 - dist2) > tolerance:
        return False, None, None
    if to_atom1_position_name != to_atom2c_position_name or from_atom1_position_name != from_atom2c_position_name:
        return False, None, None

    # ==============================================================================
    # STEP 2: Handle Edge Case (Self-Hopping / On-Site Terms)
    # ==============================================================================
    is_self_hopping = (dist1 < tolerance)
    if is_self_hopping:
        # # For self-hopping, center and neighbor are the same atom
        ## We need to check if the operation maps atom1 to atom2
        # Equation: R @ pos_atom1 + t + n_vec·basis = pos_atom2
        pos_atom1 = to_atom1.cart_coord
        pos_atom2c = to_atom2c.cart_coord
        for op_idx in range(len(magnetic_space_group_cart_spatial)):
            R, t = get_rotation_translation(magnetic_space_group_cart_spatial, op_idx)
            # Apply operation to atom1 position
            transformed_pos = R @ pos_atom1 + t
            # Calculate required shift to reach atom2
            required_lattice_shift = pos_atom2c - transformed_pos
            is_lattice, n_vec = is_lattice_vector(
                required_lattice_shift,
                lattice_basis,
                tolerance
            )
            if is_lattice:
                return True, op_idx, n_vec.astype(int)

        return False, None, None
    # ==============================================================================
    # STEP 3: Standard Case (Inter-atomic Hopping)
    # ==============================================================================
    # Displacement vector for hopping1
    for op_idx in range(len(magnetic_space_group_cart_spatial)):
        # Extract rotation R and translation t from space group operation
        R, t = get_rotation_translation(magnetic_space_group_cart_spatial, op_idx)
        # Apply rotation and translation to to_atom1.cart_coord
        # transformed = R @ r1 + t
        transformed_to_atom1_pos = R @ to_atom1.cart_coord + t
        # Calculate required lattice shift
        # match to_atom1 with to_atom2c
        required_lattice_shift = to_atom2c.cart_coord - transformed_to_atom1_pos
        # Check if required_lattice_shift is a lattice vector
        is_lattice, n_vec = is_lattice_vector(
            required_lattice_shift,
            lattice_basis,
            tolerance
        )
        if is_lattice:
            # check if this operation maps  from_atom1 to from_atom2c
            from_atoms_match = apply_full_transformation_and_check_position(from_atom1, from_atom2c, R, t,
                                                                            lattice_basis,
                                                                            n_vec,
                                                                            tolerance)
            if  from_atoms_match:
                return True, op_idx, n_vec.astype(int)
            else:
                continue
    # ==============================================================================
    # No linear relationship found
    # ==============================================================================
    return False, None, None


def add_to_root_linear(root1, root2, magnetic_space_group_cart_spatial,
                       lattice_basis, type_linear, tolerance=1e-3):
    """
    Attempt to graft root2 onto root1 as a linear child if a symmetry relationship exists.
    This function checks if root2's hopping can be generated from root1's hopping
    by applying a magnetic space group spatial part operation (rotation + translation + lattice shift).
    If a valid linear relationship is found, root2 is attached to root1 in the
    constraint tree.

    Physical Meaning:
    ----------------
    If successful, the hopping matrix T2 (of root2) is constrained by T1 (of root1):
    (i) for delta=1, T2 = [V1(g)⊗U(g)] @ T1 @ [V2(g)† ⊗ U(g)†]
    (ii) for delta=-1, T2=[V1(g)⊗~U(g)] @ T1* @ [V2(g)†⊗~U(g)†]

    Args:
        root1: First root vertex (parent candidate).
        root2: Second root vertex (child candidate).
        magnetic_space_group_cart_spatial:

        lattice_basis: 3x3 array of lattice basis vectors (each row is a basis vector)
        type_linear: String identifier for linear constraint type, value: "linear".
        tolerance: Numerical tolerance for comparison (default: 1e-3).

    Returns:
        bool: True if root2 was successfully grafted as a linear child of root1.
              False otherwise.
    Side Effects:
    -------------
    If returns True:
    - root1.children gains root2
    - root2.parent becomes root1
    - root2.is_root becomes False
    - root2.type becomes type_linear
    - root2.hopping.operation_idx and n_vec are updated to reflect the symmetry transform.
    """
    hopping1 = root1.hopping
    hopping2 = root2.hopping
    # check if hopping2 can be obtained linearly from hopping1
    # This verifies: R @ r1 + t + n_vec·basis = r2
    is_linear, op_idx, n_vec = check_hopping_linear(
        hopping1, hopping2,
        magnetic_space_group_cart_spatial,
        lattice_basis, tolerance
    )
    if is_linear == True:
        # ======================================================================
        #Perform Grafting
        # ======================================================================
        # 1. Add root2 as root1's child (updates root1.children and root2.parent)
        root1.add_child(root2)
        # 2. Update root2 properties to reflect its dependent status
        root2.type = type_linear
        root2.is_root = False
        # root2.parent is already set by add_child,
        # 3. Store the symmetry parameters required to generate T2 from T1
        root2.hopping.operation_idx = op_idx
        root2.hopping.n_vec = deepcopy(n_vec)
        return True
    else:
        return False



def add_to_root_hermitian(root1, root2, magnetic_space_group_cart_spatial,
                          lattice_basis, type_hermitian, tolerance=1e-3):
    """
    If root2's hopping is hermitian conjugate of root1's hopping,
    add root2 as root1's child with hermitian constraint. This function checks if root2 is the Hermitian conjugate of root1 under
    some magnetic space group spatial part operation. If so, it adds root2 as a child of root1 with
    the specified hermitian type and updates root2's properties accordingly.
    Args:
        root1:  First root vertex (parent)
        root2:  Second root vertex (candidate hermitian conjugate)
        magnetic_space_group_cart_spatial:  List of magnetic space group spatial part matrices in Cartesian coordinates
        lattice_basis: Primitive lattice basis vectors (3×3 array)
        type_hermitian:  String identifier for hermitian constraint type (e.g., "hermitian")
        tolerance: Numerical tolerance for comparison (default: 1e-3)

    Returns:
        bool: True if root2 was added as hermitian child of root1, False otherwise

    Example:
        For hBN:
        root1: N[0,0,0] ← B[0,0,0]
        root2: B[0,0,0] ← N[0,0,0]
        add_to_root_hermitian(root1, root2, ..., "hermitian") will add root2
        as hermitian child of root1 with type="hermitian"
    """
    hopping1 = root1.hopping
    hopping2 = root2.hopping
    # Check if hopping2 is hermitian conjugate of hopping1
    is_hermitian, op_idx, n_vec = check_hopping_hermitian(
        hopping1, hopping2, magnetic_space_group_cart_spatial,
        lattice_basis, tolerance)
    if is_hermitian == True:
        # Add root2 as root1's child
        root1.add_child(root2)
        # Set root2 properties for hermitian conjugate relationship
        root2.type = type_hermitian
        root2.is_root = False
        root2.hopping.operation_idx = op_idx
        root2.hopping.n_vec = deepcopy(n_vec)
        return True
    else:
        return False


def convert_equivalence_classes_to_hoppings(equivalence_classes, center_atom,
                                            magnetic_space_group_cart_spatial,
                                            spinor_mat_representation,delta_vec, identity_idx):
    """
    Convert all equivalence classes of neighbors into hopping objects.
    Each equivalence class contains symmetry-equivalent neighbors at the same distance.
    This function:
    1. Sorts equivalence classes by distance (nearest neighbors first)
    2. Converts each equivalence class into an equivalent hopping class

    An equivalent hopping class contains all hoppings (center ← neighbor) that are
    related by symmetry operations. All hoppings in one class have:
    - Same hopping distance
    - Same center and neighbor atom types
    - Hopping matrices related by symmetry transformations

    IMPORTANT: Returns deep copy for complete independence.
    The hopping objects themselves don't contain tree structure - that comes later
    when vertices are created with parent-child references.

    Args:
        equivalence_classes: List of equivalence classes (unsorted)
                            Each class is a list of tuples:
                            (neighbor_atom, operation_idx, n_vec)
        center_atom: atomIndex object for the center atom (hopping destination)
        magnetic_space_group_cart_spatial:  List of magnetic space group spatial part matrices in Cartesian coordinates
                                using cif origin (shape: num_ops × 3 × 4)
        spinor_mat_representation: spinor part matrices
        delta_vec: indicating time reversal, values are ±1
        identity_idx: Index of the identity operation

    Returns:
        Deep copy of list of equivalent hopping classes (sorted by distance):
        - Outer list: one equivalent hopping class per equivalence class
        - Inner list: all equivalent hoppings in that class

         Structure:
         [
            [hop_seed, hop_derived0, hop_derived1, ...],  # Class 0 (nearest, usually self)
            [hop_seed, hop_derived0, ...],                # Class 1 (next-nearest)
            ...
        ]
        Each hopping class contains:
        - One seed hopping (is_seed=True, operation_idx=identity_idx)
        - Multiple derived hoppings (is_seed=False, related by symmetry)

        Deep Copy Strategy:
        ------------------
        Returns deepcopy(all_hopping_classes) for complete independence.

         HOPPING vs VERTEX separation:
         - hopping objects: Store physical data (atoms, distance, operation_idx, etc.)
                          Can be freely copied - no tree structure inside
         - vertex objects: Store tree relationships (parent, children, is_root)
                         These will be created LATER and should NOT be deep copied
                         once the tree is built (would break parent-child references)

    """
    # ==============================================================================
    # STEP 1: Sort equivalence classes by distance
    # ==============================================================================
    # Sort by distance to center atom (nearest neighbors first)
    # Each equivalence class eq_class is a list of tuples: (neighbor_atom, operation_idx, n_vec)
    # We extract the first neighbor from each class to compute its distance
    equivalence_classes_sorted = sorted(
        equivalence_classes,
        key=lambda eq_class: np.linalg.norm(
            eq_class[0][0].cart_coord - center_atom.cart_coord, ord=2
        )
    )
    # eq_class[0][0] breakdown:
    # eq_class[0] = first tuple in the equivalence class: (neighbor_atom, operation_idx, n_vec)
    # eq_class[0][0] = neighbor_atom (first element of that tuple)
    # All members in an equivalence class have the same distance, so we use the first one

    # ==============================================================================
    # STEP 2: Convert each equivalence class to equivalent hopping class
    # ==============================================================================
    all_hopping_classes = []
    for class_id, eq_class in enumerate(equivalence_classes_sorted):
        # Convert this equivalence class to equivalent hopping class
        equivalent_hoppings =equivalent_class_to_hoppings(
            one_equivalent_class=eq_class,
            center_atom=center_atom,
            magnetic_space_group_cart_spatial=magnetic_space_group_cart_spatial,
            spinor_mat_representation=spinor_mat_representation,
            delta_vec=delta_vec,
            identity_idx=identity_idx

        )
        all_hopping_classes.append(equivalent_hoppings)

    # ==============================================================================
    # Return deep copy for complete independence
    # ==============================================================================
    # Safe to deep copy hopping objects:
    # - hopping class stores only physical data (atoms, distances, operations)
    # - No tree structure is embedded in hopping objects
    # - Tree structure lives in vertex objects (created later)
    # - Vertices wrap hoppings and add parent/children/is_root attributes
    return deepcopy(all_hopping_classes)


def construct_all_roots_for_1_atom(equivalent_hoppings_all_for_1_atom,identity_idx,type_linear):
    """
    Construct constraint tree roots for all hopping classes of one center atom.
    This function processes all equivalent hopping classes for a single center atom
    and builds a constraint tree for each class. Each tree has:
    - Root vertex: seed hopping (identity operation)
    - Children vertices: derived hoppings (symmetry operations)
    Args:
        equivalent_hoppings_all_for_1_atom:  List of hopping classes for one center atom
                                           Each element is a list of equivalent hoppings
                                           Structure: [[class_0_hoppings], [class_1_hoppings], ...]
        identity_idx:  Index of the identity operation in space_group_cart
        type_linear:  String identifier for linear constraint type (string "linear")


    Returns:
        list: List of root vertex objects, one for each hopping class
        Each root contains references to its children forming a constraint tree
    CRITICAL: Returns references, not deep copies
    --------------------------------------------
    Each root vertex in the returned list contains a tree structure with:
    - root.children = [child0, child1, ...] (references to child vertices)
    - Each child has child.parent pointing back to root
    Do NOT deep copy the returned roots - this would break tree structure!

    """
    root_list = []
    for class_idx, eq_class_hoppings in enumerate(equivalent_hoppings_all_for_1_atom):
        # Build constraint tree for this hopping class
        root = one_equivalent_hopping_class_to_root(
            eq_class_hoppings,
            identity_idx,
            type_linear
        )
        root_list.append(root)
    return root_list


def generate_all_trees_for_unit_cell(unit_cell_atoms,all_neighbors,magnetic_space_group_cart_spatial,spinor_mat_representation,delta_vec,identity_idx,type_linear,tolerance=1e-3):
    """
    Generate all trees for all atoms in the unit cell, based on equivalent neighbors around the center atom
    This function generates trees, for later tree grafting

    This function is the 1st main step that builds a complete "forest" of symmetry
    constraint trees for the entire unit cell [0,0,0].  Each tree represents one equivalence class of hoppings with
    the same center atom (hopping destination). The trees are initially  independent and will later be connected via tree graftings.


    Overview:
    ---------
    For each atom in the unit cell (center atom, hopping destination):
    1. Find all neighboring atoms within the cutoff radius
    2. Partition neighbors into equivalence classes based on symmetry
    3. Convert each equivalence class into hopping objects (center ← neighbor)
    4. Build a constraint tree for each equivalence class:
        - Root: seed hopping (generated by identity operation)
        - Children: derived hoppings (generated by other symmetry operations)
    5. Collect all trees into a single forest (a list of trees)

    The resulting forest contains trees built on symmetry around a center atom. After this function returns,
    there are two grafting procedures that find dependence between tree roots.

    In tight-binding models, the Hamiltonian matrix contains hopping terms T(i ← j)
    representing electron hopping from orbital j to orbital i. Crystal symmetry
    dramatically reduces the number of independent hopping parameters via two mechanisms
    (a) magnetic space group symmetry
    (b) Hermiticity

    Tree Structure (Before Grafting):
    ---------------------------------
    Each constraint tree in the returned forest has this structure:
        Root Vertex (seed hopping, identity operation, is_root=True)
         │
         ├── Child 0 (linear constraint, symmetry operation 1, type="linear")
         ├── Child 1 (linear constraint, symmetry operation 2, type="linear")
         ├── Child 2 (linear constraint, symmetry operation 3, type="linear")
         └── Child 3 (linear constraint, symmetry operation 4, type="linear")
         └── Child 4 (linear constraint, symmetry operation 5, type="linear")

    The root contains the independent hopping matrix (free parameters, determined by root stabilizers, this will be computed after tree graftings).
    Each child's matrix is determined by applying a symmetry transformation:
        (i) for delta=1, no time reversal,
            T_child = [V1(g)⊗U(g)] @ T_root @ [V2(g)†⊗U(g)†]
        (ii) for delta=-1, there is time reversal
             T_child =[V1(g)⊗~U(g)] @ T_root* @ [V2(g)†⊗~U(g)†]
    where V1(g) is the orbital representations of symmetry operation g, for center atom (destination)
          V2(g) is the orbital representations of symmetry operation g, for neighbor atom (source)
          U(g) is the spinor representation for g
    Args:
        unit_cell_atoms (list): List of atomIndex objects representing all atoms
                                in the reference unit cell [0,0,0]. Each atomIndex contains:
                                - Position (cell indices, fractional/Cartesian coordinates)
                                - Atom type, Wyckoff position and orbital information
                                - Pre-computed orbital representation matrices
        all_neighbors (dict): Dictionary mapping center atom index to its neighbors.
                                Format: {center_idx: [neighbor1, neighbor2, ...], ...}
                                Each value is a list of atomIndex objects within cutoff radius.
        magnetic_space_group_cart_spatial (list of np arrays): Magnetic space group spatial part operations in  Cartesian coordinates using cif origin.
                                                     Each operation is a 3×4 matrix [R|t] where:
                                                     - R (3×3): Rotation/reflection matrix
                                                     - t (3×1): Translation vector
        identity_idx: Index of the identity operation in magnetic_space_group_cart_spatial.
                      The identity operation E = [I|0] has:
                      - R = 3×3 identity matrix
                      - t = zero vector
                      Used to identify seed hoppings (roots of constraint trees).
        type_linear (str): String identifier for linear constraint type.
                            value: "linear"
                            Applied to child vertices derived from parent via symmetry operations.
                            Leads to the constraint:
                                (i) for delta=1, no time reversal,
                                    T_child = [V1(g)⊗U(g)] @ T_root @ [V2(g)†⊗U(g)†]
                                (ii) for delta=-1, there is time reversal
                                      T_child =[V1(g)⊗~U(g)] @ T_root* @ [V2(g)†⊗~U(g)†]
                            where V1(g) and V2(g) are orbital representations of operation g, U(g) is spinor
                            representation of g
    Returns:
        list: Forest of  root vertex objects (constraint tree roots).
        Each element is a vertex object representing the root of one constraint tree.
        Structure: [root_0, root_1, root_2, ..., ]

        Each root vertex contains:
        - root.hopping: The seed hopping object (center ← neighbor)
        - root.children: List of child vertex objects (derived hoppings)
        - root.parent: None (roots have no parent before grafting)
        - root.is_root: True (before grafting)
        - root.type: None (no parent constraint before grafting)
        The list is sorted for each atom center, but atom centers are not sorted

        IMPORTANT: Returns REFERENCES, not copies. Essential for tree grafting!

        Notes:
        ------
        - This function only encodes magnetic space group spatial part symmetry constraints around each center atom
        - Additional constraints (magnetic space group symmetry, Hermiticity) between roots will be dealt with
           later via tree graftings
        - Trees are built using REFERENCES, not deep copies (essential for grafting)
    """
    # ==============================================================================
    # Initialize forest of constraint trees
    # ==============================================================================
    roots_all = []
    # ==============================================================================
    # Main loop: Process each atom in the unit cell [0,0,0]
    # ==============================================================================
    for i, center_atom_i in enumerate(unit_cell_atoms):
        # ==============================================================================
        # STEP 1: Partition neighbors into equivalence classes based on symmetry
        # ==============================================================================
        equivalence_classes_center_atom_i = get_equivalent_sets_for_one_center_atom(
            center_atom_idx=i,
            unit_cell_atoms=unit_cell_atoms,
            all_neighbors=all_neighbors,
            magnetic_space_group_cart_spatial=magnetic_space_group_cart_spatial,
            identity_idx=identity_idx,
            tolerance=tolerance
        )
        # ==============================================================================
        # STEP 2: Convert equivalence classes to hopping objects
        # ==============================================================================
        equivalent_classes_hoppings_for_center_atom_i=convert_equivalence_classes_to_hoppings(
            equivalence_classes=equivalence_classes_center_atom_i,
            center_atom=center_atom_i,
            magnetic_space_group_cart_spatial=magnetic_space_group_cart_spatial,
            spinor_mat_representation=spinor_mat_representation,
            delta_vec=delta_vec,
            identity_idx=identity_idx
        )
        # ==============================================================================
        # STEP 3: Build constraint trees for each equivalence class
        # ==============================================================================
        roots_for_center_atom_i=construct_all_roots_for_1_atom(
            equivalent_hoppings_all_for_1_atom=equivalent_classes_hoppings_for_center_atom_i,
            identity_idx=identity_idx,
            type_linear=type_linear
        )
        # ==============================================================================
        # STEP 4: Add trees from this center atom to the global forest
        # ==============================================================================
        roots_all.extend(roots_for_center_atom_i)

    return roots_all


def grafting_to_existing_linear(roots_grafted_linear,root_to_be_grafted,magnetic_space_group_cart_spatial,lattice_basis,type_linear,tolerance=1e-3):
    """
    Attempt to graft a new tree onto an existing collection of  trees, as linear child
    This function checks if `root_to_be_grafted` is related by a magnetic space group spatial part symmetry
    operation to any root already in the `roots_grafted_linear` collection. If a
    linear relationship is found, the new tree is grafted onto the matching root
    as a linear child, making it dependent.

    Grafting Strategy:
    -----------------
    This function implements an "early exit" strategy:
    - Iterate through existing linear-grafted roots.
    - Check each one for a linear symmetry relationship with the new tree.
    - On the first match, graft and immediately return True.
    - If no matches are found after checking all, return False.

    Use Case:
    --------
    This is called when reducing the number of independent hopping parameters.
    As each new root is encountered, we check if it is merely a symmetry copy
     of a root we have already processed.


    Args:
        roots_grafted_linear (list):  List of root vertex objects representing
                                     roots that have already been processed/accepted.
                                     IMPORTANT: Modified in-place when grafting occurs
                                     (tree structures grow, but list itself is unchanged).
        root_to_be_grafted:  New root vertex attempting to be grafted.
                                     If grafting succeeds:
                                     - Becomes a linear child of a root in roots_grafted_linear
                                     - is_root changes from True to False
                                     - type changes from None to type_linear
                                     - Entire subtree moves with it
                                     If grafting fails:
                                     - Remains independent (caller usually adds it to the list)
        magnetic_space_group_cart_spatial  (list):
        lattice_basis (np.ndarray): Primitive lattice basis vectors.
        type_linear (str): String identifier for linear constraint type ("linear").
        tolerance:  Numerical tolerance for comparisons (default: 1e-3).

    Returns:
        bool: True if root_to_be_grafted was successfully grafted onto one of the
               existing roots in roots_grafted_linear.
               False if no linear relationship found with any existing root.

    """
    # Iterate through each root that has already been accepted as independent
    for root1 in roots_grafted_linear:
        # Attempt to graft the new root onto the existing root1\
        # add_to_root_linear handles the check and the structural update if successful
        success = add_to_root_linear(
            root1=root1,
            root2=root_to_be_grafted,
            magnetic_space_group_cart_spatial=magnetic_space_group_cart_spatial,
            lattice_basis=lattice_basis,
            type_linear=type_linear,
            tolerance=tolerance
        )
        if success == True:
            # Early exit: We found a parent!
            # The tree is now grafted, so we stop searching.
            return True
    # If we finish the loop without returning, no parent was found
    return False




def tree_grafting_linear(roots_all,magnetic_space_group_cart_spatial,lattice_basis,type_linear,tolerance=1e-3):
    """
    Perform Linear tree grafting on all constraint trees.
    This function implements a symmetry reduction step based on linear constraint. It iterates through
    all root vertices and attempts to graft each one onto existing trees if a linear symmetry relationship exists.

    Algorithm:
    ---------
    1. Deep copy all roots to avoid modifying the input.
    2. Initialize roots_grafted_linear with the 0th root.
    3. For each remaining root:
        a. Try to graft it onto any existing root in roots_grafted_linear using
           magnetic space group spatial part symmetry (rotation + translation + lattice shift).
        b. If grafting succeeds: the root becomes a linear child (dependent).
        c. If grafting fails: add the root to roots_grafted_linear as a new independent root.
    4. Return the final collection of independent roots.

    Tree Structure After Grafting:
    -----------------------------
    Before:
        Root A (independent)          Root B (independent)
    After (if B is symmetry equivalent to A):
        Root A
        ├── ... (existing children)
        └── Root B (linear) ← Now a child of A!
            └── ... (B's subtree moves with it)
    Physical Meaning:
    ----------------
    If root B is grafted as a linear child of root A, it implies that the hopping
    matrix represented by B is not  free, but is related to A by symmetry:
    (i) for delta=1, no time reversal,
            T(B) = [V1(g)⊗U(g)] @ T(A) @ [V2(g)†⊗U(g)†]
        (ii) for delta=-1, there is time reversal
            T(B) = [V1(g)⊗~U(g)] @ T(A)* @ [V2(g)†⊗~U(g)†]
    Args:
        roots_all (list): List of root vertex objects
        magnetic_space_group_cart_spatial (list): Magnetic space group spatial part operations in Cartesian coordinates.
        lattice_basis (np.ndarray): Primitive lattice basis vectors.
        type_linear (str): String identifier for linear constraint type ("linear").
        tolerance (float): Numerical tolerance for comparisons (default: 1e-3).

    Returns:
        list: Collection of root vertex objects after Linear grafting.

    """
    # ==============================================================================
    # STEP 1: Initialize working variables
    # ==============================================================================
    roots_all_num = len(roots_all)
    # Deep copy to ensure input list remains unmodified
    roots_all_copy = deepcopy(roots_all)
    # Initialize the list of independent roots with the 0th one
    roots_grafted_linear = [roots_all_copy[0]]
    # ==============================================================================
    # STEP 2: Iterate through remaining roots and attempt grafting
    # ==============================================================================
    for j in range(1, roots_all_num):
        root_to_be_grafted = roots_all_copy[j]
        # Attempt to graft onto existing independent roots
        was_grafted = grafting_to_existing_linear(
            roots_grafted_linear=roots_grafted_linear,
            root_to_be_grafted=root_to_be_grafted,
            magnetic_space_group_cart_spatial=magnetic_space_group_cart_spatial,
            lattice_basis=lattice_basis,
            type_linear=type_linear,
            tolerance=tolerance
        )
        if was_grafted == True:
            pass
        else:
            # If no relationship found, this root remains independent
            roots_grafted_linear.append(root_to_be_grafted)

    return roots_grafted_linear
































































tol=1e-3
roots_from_eq_class=generate_all_trees_for_unit_cell(unit_cell_atoms,all_neighbors,magnetic_space_group_cart_spatial,spinor_mat_representation,delta_vec,identity_idx,type_linear,tol)
# print_all_trees(roots_from_eq_class)
#########################find linear first, hermitian second
roots_grafted_linear=tree_grafting_linear(roots_from_eq_class,
                                          magnetic_space_group_cart_spatial,
                                          lattice_basis,
                                          type_linear,tol)


# print_all_trees(roots_grafted_linear)