import re
import sys
import os
import pickle
import numpy as np
from pathlib import Path
import copy


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from name_conventions import symmetry_matrices_file_name
#this script parse .cif file to get atom positions, symmetry operations
#.cif file is generated from https://iso.byu.edu/findsym.php


paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist
tol=1e-3
if len(sys.argv) != 2:
    print("wrong number of arguments.", file=sys.stderr)
    print("usage: python parse_cif.py /path/to/xxx.cif", file=sys.stderr)
    exit(paramErrCode)



cif_file_name = str(sys.argv[1])
# Check if configuration file exists
if not os.path.exists(cif_file_name):
    print(f"file not found: {cif_file_name}", file=sys.stderr)
    exit(fileNotExistErrCode)


def remove_comments_and_empty_lines_cif(file):
    """
    Remove comments and empty lines from a CIF-like configuration file.

    - Comments start with # and continue to end of line.
    - # inside single (') or double (") quotes are preserved.
    - Empty lines (or lines with only whitespace) are removed.

    :param file: conf file path
    :return: list of cleaned lines
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    # Regex explanation:
    # 1. (["\'])  -> Capture group 1: Match a quote (single or double)
    # 2. (?:      -> Non-capturing group for content inside quotes
    #    \\.      -> Match escaped characters (like \")
    #    |        -> OR
    #    [^\\]    -> Match any character that isn't a backslash
    #    )*?      -> Match as few times as possible (lazy)
    # 3. \1       -> Match the closing quote (same as group 1)
    # 4. |        -> OR
    # 5. (#.*$)   -> Capture group 2: Match a real comment (starts with #, goes to end)
    # This pattern finds quoted strings OR comments.
    # We will use a callback to decide what to do.
    pattern = re.compile(r'(["\'])(?:\\.|[^\\])*?\1|(#.*$)')
    for oneLine in lines:
        # We use a callback function for re.sub
        # If group 2 (the comment) exists, replace it with empty string.
        # If group 1 (the quote) matches, return the whole match (preserve it).
        def replace_func(match):
            if match.group(2):
                return ""  # It's a comment, remove it
            else:
                return match.group(0)  # It's a quoted string, keep it
        # Apply the regex
        cleaned_line = pattern.sub(replace_func, oneLine).strip()
        # Only add non-empty lines
        if cleaned_line:
            linesToReturn.append(cleaned_line)

    return linesToReturn



# coef_pattern = re.compile(r"([+-]?)\s*([xyz])", re.IGNORECASE)
# # Matches: optional sign, optional space, number (integer, decimal, or fraction)
# # Example: "+1/2", "-0.5", "1/4", "+1"
# translation_pattern = re.compile(r"([+-]?)\s*(\d+(?:[./]\d+)?)")


# The Master Regex
# Group 1,2: Variable (sign, char)
# Group 3,4: Number (sign, val)
term_pattern = re.compile(r"([+-]?)\s*([xyz])|([+-]?)\s*(\d+(?:[./]\d+)?)", re.IGNORECASE)

def parse_single_expression(expression_str):
    """
    Parses a string like '-x+1/2' or 'x-y' into numerical coefficients.
    :param expression_str:
    :return: (coeff_x, coeff_y, coeff_z, translation)
    """
    # Initialize values
    c_x = 0.0
    c_y = 0.0
    c_z = 0.0
    trans = 0.0
    # Find all matches in the string
    for match in term_pattern.finditer(expression_str):
        # --- CASE 1: It's a Variable: x, y, z ---
        if match.group(2):
            sign_str = match.group(1)
            variable = match.group(2).lower()
            # Determine value: +1 or -1
            value = -1.0 if sign_str == '-' else 1.0
            if variable == 'x':
                c_x += value
            elif variable == 'y':
                c_y += value
            elif variable == 'z':
                c_z += value
        # --- CASE 2: It's a Number: Translation ---
        elif match.group(4):
            sign_str = match.group(3)
            number_str = match.group(4)
            # Handle fraction conversion (e.g., "1/2" -> 0.5)
            if '/' in number_str:
                num, den = number_str.split('/')
                val = float(num) / float(den)
            else:
                val = float(number_str)
            # Apply sign
            if sign_str == '-':
                val = -val
            trans += val

    return c_x, c_y, c_z, trans


# # --- TEST EXAMPLES ---
# examples = [
#     "-x+1/2",       # Standard
#     "1/2-x",        # Non-standard (Translation first)
#     "0.5+y",        # Non-standard (Decimal + Variable)
#     "z",            # Simple
#     "-y-x+1/4"      # Complex
#     "-1",
#     "1"
# ]
#
# print(f"{'Expression':<15} | {'x':<4} {'y':<4} {'z':<4} {'Trans':<5}",file=sys.stdout)
# print("-" * 45,file=sys.stdout)
#
# for ex in examples:
#     cx, cy, cz, tr = parse_single_expression(ex)
#     print(f"{ex:<15} | {cx:<4} {cy:<4} {cz:<4} {tr:<5}",file=sys.stdout)


def parse_cif_contents_xyz_transformations(file):
    """
    Parses the CIF file to extract symmetry operations, including magnetic time-reversal.
    Returns a list of dictionaries, each containing:
      - 'matrix': A list of 3 dictionaries (for x', y', z' components).
      - 'delta': The time-reversal symmetry component (+1.0 or -1.0). Defaults to 1.0.
    """
    # Get cleaned lines from file
    lines = remove_comments_and_empty_lines_cif(file)

    symmetry_operations = []

    # State variables for parsing
    in_loop = False
    current_loop_headers = []

    for line in lines:
        line = line.strip()

        # 1. Check for start of a loop
        if line == "loop_":
            in_loop = True
            current_loop_headers = []
            continue

        # 2. Check for headers (lines starting with underscore)
        if line.startswith("_"):
            if in_loop:
                current_loop_headers.append(line)
            else:
                in_loop = False
                current_loop_headers = []
            continue

        # 3. Process Data Lines
        # Check if we are in a loop containing either standard or magnetic symmetry operations
        has_magn_xyz = "_space_group_symop_magn_operation.xyz" in current_loop_headers
        has_std_xyz = "_symmetry_equiv_pos_as_xyz" in current_loop_headers

        if in_loop and (has_magn_xyz or has_std_xyz):
            # Identify which header we are using
            xyz_header = "_space_group_symop_magn_operation.xyz" if has_magn_xyz else "_symmetry_equiv_pos_as_xyz"
            xyz_index = current_loop_headers.index(xyz_header)

            # Split line by whitespace to handle optional columns like .id
            parts = line.split()

            # Extract the xyz string based on its column index
            if len(parts) > xyz_index:
                xyz_str = parts[xyz_index]
            else:
                xyz_str = line  # Fallback if formatting is unusual

            # Remove quotes
            clean_line = xyz_str.replace("'", "").replace('"', "")

            # Split into components (x, y, z, and optionally delta)
            raw_components = clean_line.split(",")

            if len(raw_components) >= 3:
                op_matrix = []
                # Parse the first 3 components (spatial: x, y, z)
                for comp in raw_components[:3]:
                    cx, cy, cz, tr = parse_single_expression(comp.strip())
                    op_matrix.append({
                        "raw_string": comp.strip(),
                        "cx": cx,
                        "cy": cy,
                        "cz": cz,
                        "trans": tr
                    })

                # Parse the 4th component (time-reversal: delta) if it exists
                delta = 1.0
                if len(raw_components) == 4:
                    try:
                        delta = float(raw_components[3].strip())
                    except ValueError:
                        pass  # Default to 1.0 if parsing fails

                # Store both the spatial matrix and the time-reversal delta
                symmetry_operations.append({
                    "matrix": op_matrix,
                    "delta": delta
                })
            else:
                print(f"Warning: Could not parse symmetry line: {line}", file=sys.stderr)

    return symmetry_operations


def parse_cell_parameters(file):
    """
    Parses unit cell lengths and angles from a CIF file.
    Returns a dictionary containing a, b, c, alpha, beta, gamma.
    Raises ValueError if any parameter is missing.
    """
    # Get cleaned lines
    lines = remove_comments_and_empty_lines_cif(file)

    # Initialize dictionary with None to detect missing values later
    cell_params = {
        'a': None,
        'b': None,
        'c': None,
        'alpha': None,
        'beta': None,
        'gamma': None
    }

    # Map CIF keywords to our dictionary keys
    keyword_map = {
        '_cell_length_a': 'a',
        '_cell_length_b': 'b',
        '_cell_length_c': 'c',
        '_cell_angle_alpha': 'alpha',
        '_cell_angle_beta': 'beta',
        '_cell_angle_gamma': 'gamma'
    }

    for line in lines:
        # Split line by whitespace
        parts = line.strip().split()

        # We expect at least "KEY VALUE" (2 parts)
        if len(parts) >= 2:
            key = parts[0]
            val_str = parts[1]

            if key in keyword_map:
                # Handle uncertainty notation often found in CIFs (e.g., "12.34(5)")
                clean_val_str = val_str.split('(')[0]

                try:
                    value = float(clean_val_str)
                    target_key = keyword_map[key]
                    cell_params[target_key] = value
                except ValueError:
                    # We raise an error here immediately if the number format is wrong
                    raise ValueError(f"Error parsing numerical value for {key}: '{val_str}' in file {file}")

    # --- CHECK FOR MISSING VALUES ---
    missing_keys = [k for k, v in cell_params.items() if v is None]

    if missing_keys:
        # Raise an error stopping execution if parameters are missing
        raise ValueError(f"Missing required cell parameters in {file}: {', '.join(missing_keys)}")

    return cell_params


def parse_atom_sites(file):
    """
    Parses the atom site loop to extract fractional coordinates and labels.
    Returns a list of dictionaries.

    Required headers:
      - _atom_site_label
      - _atom_site_type_symbol
      - _atom_site_fract_x
      - _atom_site_fract_y
      - _atom_site_fract_z

    Ignores extra columns such as symmetry_multiplicity, Wyckoff_symbol, and fract_symmform.
    """
    lines = remove_comments_and_empty_lines_cif(file)

    atoms = []

    # State variables
    in_loop = False
    loop_headers = []

    # Define mandatory headers
    required_headers = [
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z"
    ]

    for line in lines:
        line = line.strip()

        # 1. Detect Loop Start
        if line == "loop_":
            in_loop = True
            loop_headers = []
            continue

        # 2. Collect Headers
        if line.startswith("_"):
            if in_loop:
                loop_headers.append(line)
            else:
                in_loop = False
                loop_headers = []
            continue

        # 3. Process Data Lines
        # We identify the atom loop by checking if it contains one of our unique keys (e.g., fract_x)
        if in_loop and "_atom_site_fract_x" in loop_headers:

            # --- VALIDATION: Check for Missing Headers ---
            missing_headers = [h for h in required_headers if h not in loop_headers]
            if missing_headers:
                raise ValueError(f"Missing mandatory headers in atom site loop: {', '.join(missing_headers)}")

            # Split the line by whitespace into a list of values
            parts = line.split()

            # If the line is empty or doesn't have enough parts, skip or break
            if not parts:
                continue

            atom_data = {}

            try:
                # --- Map Headers to Indices ---
                # This dynamically finds which column corresponds to which piece of data.
                # By doing this, we naturally ignore columns like _atom_site_Wyckoff_symbol.
                idx_lbl = loop_headers.index("_atom_site_label")
                idx_sym = loop_headers.index("_atom_site_type_symbol")
                idx_x = loop_headers.index("_atom_site_fract_x")
                idx_y = loop_headers.index("_atom_site_fract_y")
                idx_z = loop_headers.index("_atom_site_fract_z")

                # --- Validate Column Count ---
                max_idx = max(idx_lbl, idx_sym, idx_x, idx_y, idx_z)
                if len(parts) <= max_idx:
                    # If we hit a line that doesn't have enough columns, it might be the end of the loop
                    continue

                # --- Extract Strings ---
                atom_data['label'] = parts[idx_lbl]
                atom_data['symbol'] = parts[idx_sym]

                # --- Extract Floats (Handle uncertainty like '0.123(4)') ---
                atom_data['x'] = float(parts[idx_x].split('(')[0])
                atom_data['y'] = float(parts[idx_y].split('(')[0])
                atom_data['z'] = float(parts[idx_z].split('(')[0])

                # Optional: Extract Occupancy and Uiso if present (not mandatory for error)
                if "_atom_site_occupancy" in loop_headers:
                    idx_occ = loop_headers.index("_atom_site_occupancy")
                    if idx_occ < len(parts):
                        atom_data['occupancy'] = float(parts[idx_occ].split('(')[0])

                if "_atom_site_U_iso_or_equiv" in loop_headers:
                    idx_u = loop_headers.index("_atom_site_U_iso_or_equiv")
                    if idx_u < len(parts):
                        atom_data['u_iso'] = float(parts[idx_u].split('(')[0])

                atoms.append(atom_data)

            except ValueError as e:
                raise ValueError(f"Error parsing atom data in line: '{line}'. Reason: {e}")

    # --- FINAL CHECK ---
    if not atoms:
        raise ValueError("No atom sites found. The file may be missing the atom loop or the loop is empty.")

    return atoms


def parse_symmetry_metadata(file_path):
    """
    Parses the CIF file to extract standard and magnetic space group metadata:
    1. The Data Block Name (e.g., Te2W)
    2. Space Group Name (H-M)
    3. Int Tables Number
    4. Cell Setting
    5. Magnetic Space Group metadata (BNS, OG, UNI, Litvin)
    """
    lines = remove_comments_and_empty_lines_cif(file_path)

    # Initialize dictionary with None to track missing values if needed
    metadata = {
        'data_name': None,
        'space_group_name_H_M': None,
        'int_tables_number': None,
        'cell_setting': None,
        # --- New Magnetic Metadata Fields ---
        'space_group_magn_number_BNS': None,
        'space_group_magn_name_UNI': None,
        'space_group_magn_name_BNS': None,
        'space_group_magn_number_OG': None,
        'space_group_magn_name_OG': None,
        'point_group_number_Litvin': None,
        'point_group_name_UNI': None
    }

    for line in lines:
        # 1. Extract Data Block Name (starts with 'data_')
        if line.startswith("data_"):
            # Everything after "data_" is the name
            metadata['data_name'] = line[5:].strip()
            continue

        # For the other fields, we split the line into Key and Value.
        # maxsplit=1 ensures we only split on the first whitespace, preserving spaces in the value if any.
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue

        key = parts[0]
        value = parts[1]

        # Remove single quotes (') and double quotes (") from the value
        clean_value = value.strip("'\"")

        # --- Standard Space Group ---
        if key == "_symmetry_space_group_name_H-M":
            metadata['space_group_name_H_M'] = clean_value
        elif key == "_symmetry_Int_Tables_number":
            try:
                metadata['int_tables_number'] = int(clean_value)
            except ValueError:
                metadata['int_tables_number'] = clean_value  # Keep as string if conversion fails
        elif key == "_symmetry_cell_setting":
            metadata['cell_setting'] = clean_value

        # --- Magnetic Space Group ---
        elif key == "_space_group_magn.number_BNS":
            metadata['space_group_magn_number_BNS'] = clean_value
        elif key == "_space_group_magn.name_UNI":
            metadata['space_group_magn_name_UNI'] = clean_value
        elif key == "_space_group_magn.name_BNS":
            metadata['space_group_magn_name_BNS'] = clean_value
        elif key == "_space_group_magn.number_OG":
            metadata['space_group_magn_number_OG'] = clean_value
        elif key == "_space_group_magn.name_OG":
            metadata['space_group_magn_name_OG'] = clean_value
        elif key == "_space_group_magn.point_group_number_Litvin":
            metadata['point_group_number_Litvin'] = clean_value
        elif key == "_space_group_magn.point_group_name_UNI":
            metadata['point_group_name_UNI'] = clean_value

    return metadata


def metadata_to_key_value(metadata):
    """

    Args:
        metadata: output from parse_symmetry_metadata

    Returns: dictionary for constructing .conf file

    """
    out_dict={}
    for key, value in metadata.items():
        if key=="data_name":
            out_dict["name"]=value
        else:
            out_dict[key]=value

    return out_dict

def parse_transformation_one_expression_to_vector(one_transformation_dict):
    """
    Args:
        one_transformation_dict: a dict represents one expression in a row of symmetry transformations

    Returns:  a row of vector, length=4, first 3 numbers are coefficients for rotation matrix,
                the last number is for translation
    """
    ret_vec = np.zeros(4)
    for key, value in one_transformation_dict.items():
        if key == "cx":
            ret_vec[0] = value
        elif key == "cy":
            ret_vec[1] = value
        elif key == "cz":
            ret_vec[2] = value
        elif key == "trans":
            ret_vec[3] = value
    return ret_vec


def parse_transformation_one_row_to_matrix(symmetry_op_dict):
    """
    Args:
        symmetry_op_dict: a dict containing "matrix" (list of 3 dicts for x,y,z)
                          and "delta" (float for time-reversal).

    Returns:
        mat: 3 by 4 symmetry operation matrix
        delta: time-reversal symmetry multiplier (+1.0 or -1.0)
    """
    mat = np.zeros((3, 4))

    # Extract the list of spatial expressions and the delta value
    matrix_list = symmetry_op_dict["matrix"]
    delta = symmetry_op_dict.get("delta", 1.0)  # Default to 1.0 if delta is missing

    for row, dict_item in enumerate(matrix_list):
        ret_vec = parse_transformation_one_expression_to_vector(dict_item)
        mat[row, :] = ret_vec

    return mat, delta


def subroutine_generate_all_symmetry_transformation_matrices(cif_file_path):
    """
    This function writes the symmetry transformation matrices and delta to a pkl file.
    """
    symmetry_operations = parse_cif_contents_xyz_transformations(cif_file_path)
    out_list = []

    for counter, op_dict in enumerate(symmetry_operations):
        # Now we get both the 3x4 matrix and the delta value directly from the parser
        mat, delta = parse_transformation_one_row_to_matrix(op_dict)

        # Append as a dictionary to the list
        out_list.append({
            "mat": mat,
            "delta": delta
        })

    cif_dir = Path(cif_file_path).resolve().parent
    out_pickle_file_name = str(cif_dir / symmetry_matrices_file_name)

    # --- SAVE TO PICKLE ---
    with open(out_pickle_file_name, 'wb') as f:
        pickle.dump(out_list, f)

    return out_pickle_file_name


def generate_unit_cell_basis(cell_params,tol):
    """
    Generates basis vectors from cell parameters with defensive validation.
    Args:
        cell_params: Dictionary containing 'a', 'b', 'c' (Angstroms)
                     and 'alpha', 'beta', 'gamma' (Degrees).

    Returns:  basis_mat: 3x3 np.array where each row is a basis vector.
    Raises:
        KeyError: If keys are missing.
        ValueError: If lengths are <= 0, angles are out of bounds,
                    or angles form an impossible geometry (zero/negative volume).
    """
    # 1. Validate Keys exist
    required_keys = {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
    if not required_keys.issubset(cell_params):
        missing = required_keys - set(cell_params.keys())
        raise KeyError(f"Missing required cell parameters: {missing}")
    a = cell_params["a"]
    b = cell_params["b"]
    c = cell_params["c"]
    alpha_deg = cell_params["alpha"]
    beta_deg = cell_params["beta"]
    gamma_deg = cell_params["gamma"]
    # 2. Validate Lengths (Must be strictly positive)

    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"Lattice parameters a, b, c must be positive. Got: a={a}, b={b}, c={c}")
    # 3. Validate Angles (Must be strictly between 0 and 180 for a valid cell)
    # sin(0) or sin(180) would cause division by zero later
    if not (0 < alpha_deg < 180):
        raise ValueError(f"Alpha must be between 0 and 180 degrees. Got: {alpha_deg}")
    if not (0 < beta_deg < 180):
        raise ValueError(f"Beta must be between 0 and 180 degrees. Got: {beta_deg}")
    if not (0 < gamma_deg < 180):
        raise ValueError(f"Gamma must be between 0 and 180 degrees. Got: {gamma_deg}")

    # Convert degrees to radians
    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    gamma_rad = np.radians(gamma_deg)
    # v0 vector (aligned with x-axis)
    v0_row_vec = np.array([a, 0, 0])
    # v1 vector (in xy-plane)
    v1_row_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    # v2 vector calculation
    v2_0 = c * np.cos(beta_rad)
    # frac corresponds to (cos(alpha) - cos(beta)cos(gamma)) / sin(gamma)
    # We already validated gamma != 0 or 180, so sin(gamma) is safe.
    frac = (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    v2_1 = c * frac
    # 4. Validate Geometric Consistency (Volume Check)
    # The term under the square root determines the height of the cell (z-component).
    # If this term is negative, the angles provided cannot form a closed 3D object.
    term_under_sqrt = np.sin(beta_rad) ** 2 - frac ** 2
    # We use a small epsilon for float comparison to avoid errors on valid but edge-case cells
    epsilon = 1e-9
    if term_under_sqrt < epsilon:
        # If it's negative, the geometry is impossible.
        # If it's effectively zero, the cell is flat (2D), which is invalid for a 3D basis generator.
        raise ValueError(
            f"Invalid angular configuration: alpha={alpha_deg}, beta={beta_deg}, gamma={gamma_deg}. "
            "These angles do not form a valid 3D unit cell (volume is zero or complex)."
        )
    v2_2 = c * np.sqrt(term_under_sqrt)

    v2_row_vec = np.array([v2_0, v2_1, v2_2])
    basis_mat = np.array([v0_row_vec, v1_row_vec, v2_row_vec])
    # --- APPLY TOLERANCE CHECK ---
    # If absolute value of any component is less than tol, set it to 0.0
    basis_mat[np.abs(basis_mat) < tol] = 0.0
    return basis_mat

def rename_labels(atoms_list):
    """
    Renames atom labels based on their element symbol.
    Atoms are grouped by symbol, then numbered sequentially starting from 0.
    This function creates a DEEP COPY of the input list, so the original
    data remains unmodified.
    Args:
        atoms_list: The list of dictionaries returned by parse_atom_sites.

    Returns:
        A NEW list of dictionaries with updated 'label' keys.

    """
    # Create a deep copy so we don't mutate the original list
    atoms_copy = copy.deepcopy(atoms_list)
    # Dictionary to track the count for each element group
    # Structure will look like: {'O': 2, 'V': 1, 'Te': 1}
    symbol_counts = {}
    for atom in atoms_copy:
        # 1. Get the symbol to identify the group (e.g., 'O', 'V')
        sym = atom["symbol"]
        # 2. If this is the first time we see this symbol, initialize count to 0
        if sym not in symbol_counts:
            symbol_counts[sym] = 0
        # 3. Generate the new label using the current count
        # Example: "O" + "0" -> "O0"
        new_label = f"{sym}{symbol_counts[sym]}"
        # 4. Update the atom dictionary in the COPY
        # We preserve the old label in 'original_label' for debugging
        atom["original_label"] = atom["label"]
        atom["label"] = new_label
        # 5. Increment the counter for this specific group
        symbol_counts[sym] += 1

    return atoms_copy


def subroutine_generate_conf_file(cif_file_name, tol):
    metadata = parse_symmetry_metadata(cif_file_name)

    # --- Extract Standard Metadata ---
    data_name = metadata.get("data_name", "Unknown")
    space_group_name_H_M = metadata.get("space_group_name_H_M", "")
    int_tables_number = metadata.get("int_tables_number", "")
    cell_setting = metadata.get("cell_setting", "")

    # --- Extract Magnetic Metadata ---
    space_group_magn_number_BNS = metadata.get("space_group_magn_number_BNS", "")
    space_group_magn_name_UNI = metadata.get("space_group_magn_name_UNI", "")
    space_group_magn_name_BNS = metadata.get("space_group_magn_name_BNS", "")
    space_group_magn_number_OG = metadata.get("space_group_magn_number_OG", "")
    space_group_magn_name_OG = metadata.get("space_group_magn_name_OG", "")
    point_group_number_Litvin = metadata.get("point_group_number_Litvin", "")
    point_group_name_UNI = metadata.get("point_group_name_UNI", "")

    # --- Parse Atoms and Cell ---
    atoms_list = parse_atom_sites(cif_file_name)
    atoms_list_renamed = rename_labels(atoms_list)
    Wyckoff_position_num = len(atoms_list)

    cell_params = parse_cell_parameters(cif_file_name)
    basis_mat = generate_unit_cell_basis(cell_params, tol)
    v0, v1, v2 = basis_mat
    basis_str = f"{v0[0]}, {v0[1]}, {v0[2]}; {v1[0]}, {v1[1]}, {v1[2]}; {v2[0]}, {v2[1]}, {v2[2]}\n"

    # --- Build Configuration File Text ---
    text_list = [
        f"#This is the configuration file for {data_name} computations\n",
        "#the format is key=value\n",
        "#matches: key=value\n",
        "# my-key = value\n",
        "# my.key= value with spaces\n",
        "#KEY_NAME =   value123\n",
        "# complex-key.name = some value here\n",
        "#does not match:\n",
        "# = value              # No key\n",
        "# key =                # No value, this must be filled\n",
        "# key value            # No equals sign\n",
        "# my key = value       # Space in key (not allowed)\n",
        "# key==value          # Multiple equals (first = becomes part of key)\n",
        "\n",
        "#name of the system\n",
        f"name={data_name}\n",
        "\n",
        "#dimension of system\n",
        "dim=\n",
        "\n",
        "#directions to study, available directions x,y,z\n",
        "directions_to_study=\n",
        "\n",
        "#whether spin is considered\n",
        "spin=True\n",
        "\n",
        "truncation_radius=\n",
        "\n",
        f"lattice_basis={basis_str}\n",
        "\n",
        f"space_group={int_tables_number}\n",
        "\n",
        f"space_group_origin=0,0,0\n",
        "\n",
        f"space_group_name_H_M={space_group_name_H_M}\n",
        "\n",
        f"cell_setting={cell_setting}\n",
        "\n",
        "# --- Magnetic Space Group Metadata ---\n",
        f"space_group_magn_number_BNS={space_group_magn_number_BNS}\n",
        f"space_group_magn_name_UNI={space_group_magn_name_UNI}\n",
        f"space_group_magn_name_BNS={space_group_magn_name_BNS}\n",
        f"space_group_magn_number_OG={space_group_magn_number_OG}\n",
        f"space_group_magn_name_OG={space_group_magn_name_OG}\n",
        f"point_group_number_Litvin={point_group_number_Litvin}\n",
        f"point_group_name_UNI={point_group_name_UNI}\n",
        "\n",
        f"Wyckoff_position_num={Wyckoff_position_num}\n",
        "\n",
    ]

    # --- Append Atom Data ---
    for dict_item in atoms_list_renamed:
        label = dict_item["label"]
        x = dict_item["x"]
        y = dict_item["y"]
        z = dict_item["z"]
        str0 = f"#Wyckoff position label {label}, input orbitals\n"
        str1 = f"{label}_orbitals=\n"
        str2 = f"#one position of {label}, coefficients  (fractional coordinates)\n"
        str3 = f"{label}_position_coefs={x}, {y}, {z}\n"
        str_list = ["\n", str0, str1, str2, str3]
        text_list.extend(str_list)

    # --- Write to File ---
    out_dir = Path(cif_file_name).resolve().parent
    out_conf_file = str(out_dir / f"{data_name}.conf")
    with open(out_conf_file, 'w', encoding='utf-8') as f:
        f.writelines(text_list)
        print(f"Successfully generated configuration file: {out_conf_file}")


subroutine_generate_conf_file(cif_file_name,tol)
subroutine_generate_all_symmetry_transformation_matrices(cif_file_name)