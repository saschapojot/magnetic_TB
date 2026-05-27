import pickle
import re
import sys
from pathlib import Path

import numpy as np
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from name_conventions import BZ_path_file_name,processed_input_pkl_file_name


# ==============================================================================
# Path and Directory Utilities
# ==============================================================================
def get_data_directory(conf_file_path: str) -> str:
    """
    Extract the directory containing the configuration file and data files.
    Args:
        conf_file_path:  Path to the configuration file

    Returns:
         String path to the data directory
    Raises:
        FileNotFoundError: If configuration file doesn't exist

    """
    config_path = Path(conf_file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {conf_file_path}")

    config_dir = config_path.parent
    return str(config_dir.resolve())


def get_file_paths(data_dir: str)-> dict:
    """

    Args:
        data_dir: data directory

    Returns: dict containing BZ_path file path and preprocessed_input_file path

    """
    data_path = Path(data_dir)
    return {
        "BZ_path_file": str(data_path / BZ_path_file_name),
        "preprocessed_input_file": str(data_path/processed_input_pkl_file_name)
    }

def validate_BZ_path_file(file_paths_dict: dict) -> None:
    """
    Verify that the k-path configuration file and preprocessed input exist.
    Args:
        file_paths_dict: Dictionary containing paths to required files.

    Returns:
        None

     Raises:
        FileNotFoundError: If any of the files do not exist.
    """
    missing_files = []
    file_descriptions = {
        'BZ_path_file': "k-path endpoints",
        'preprocessed_input_file': "processed input parameters"
    }
    for key, description in file_descriptions.items():
        file_path = file_paths_dict.get(key)

        # Check if the path is None or if the file does not exist
        if not file_path or not Path(file_path).exists():
            missing_files.append(f"Missing {description} file at: {file_path}")

    if missing_files:
        error_message = "The following required files were not found:\n" + "\n".join(missing_files)
        raise FileNotFoundError(error_message)


# ==============================================================================
# Define regex patterns for parsing
# ==============================================================================
# General key=value pattern
key_value_pattern = r'^([^=\s]+)\s*=\s*([^=]*)\s*$'
# Pattern for floating point numbers (including scientific notation)
float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

#regex for fractional coordinates
# 1D: x (Single value)
fractional_coord_1d_pattern = rf"^\s*({float_pattern})\s*$"

# 2D: x, y (Comma separated)
fractional_coord_2d_pattern=rf"^\s*({float_pattern})\s*,\s*({float_pattern})\s*$"

# 3D: x, y, z (Comma separated, based on your input)
fractional_coord_3d_pattern = rf"^\s*({float_pattern})\s*,\s*({float_pattern})\s*,\s*({float_pattern})\s*$"


def removeCommentsAndEmptyLines(file):
    """
    Remove comments and empty lines from configuration file

    Comments start with # and continue to end of line
    Empty lines (or lines with only whitespace) are removed

    :param file: conf file path
    :return: list of cleaned lines (comments and empty lines removed)
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    for oneLine in lines:
        # Remove comments (everything after #) and strip whitespace
        oneLine = re.sub(r'#.*$', '', oneLine).strip()

        # Only add non-empty lines
        if oneLine:
            linesToReturn.append(oneLine)

    return linesToReturn


def parse_preprocessed_input(preprocessed_input_file_name):
    """
    Args:
        preprocessed_input_file_name: contains parsed information, is a pkl file

    Returns: a dictionary containing all parsed information
    """
    try:
        with open(preprocessed_input_file_name, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The preprocessed input file was not found at: {preprocessed_input_file_name}")
    except pickle.UnpicklingError:
        raise ValueError(f"Error decoding the pickle file: {preprocessed_input_file_name}. It may be corrupted.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading {preprocessed_input_file_name}: {e}")



def read_BZ_path_conf(BZ_path_file_name: str, processed_input_data):
    """
    Reads the k-path configuration file.
    Always expects 3D coordinates [x, y, z] in the file.
    Selects specific components based on 'directions_to_study' (e.g., if directions=['x', 'z'],
    it takes the 1st and 3rd components from the file and places them into the final vector).

    Args:
        BZ_path_file_name: Path to the BZ_path.conf file.
        processed_input_data: Dictionary containing parsed input data (must contain 'directions_to_study').

    Returns:
        list: A list of dictionaries, where each dictionary represents a k-point:
              {'label': str, 'coords': [float, float, float]}
    """
    # 1. Retrieve directions
    try:
        directions = processed_input_data["parsed_config"].get('directions_to_study', ['x', 'y', 'z'])
    except KeyError:
        raise KeyError(
            "The processed_input_data dictionary is missing 'parsed_config' or 'directions_to_study'.")

    # 2. Get cleaned lines from file
    linesWithCommentsRemoved = removeCommentsAndEmptyLines(BZ_path_file_name)

    parsed_k_points = []

    # 3. Parse each line
    for oneLine in linesWithCommentsRemoved:
        # Check if line matches key=value format
        matchLine = re.match(key_value_pattern, oneLine)

        if matchLine:
            key_label = matchLine.group(1).strip()
            value_str = matchLine.group(2).strip()

            # Initialize 3D coords with zeros
            final_coords = [0.0, 0.0, 0.0]

            # Always parse as 3D coordinate
            coord_match = re.match(fractional_coord_3d_pattern, value_str)

            if coord_match:
                # Extract the 3 raw values from the file: [val_x, val_y, val_z]
                raw_values = [
                    float(coord_match.group(1)),
                    float(coord_match.group(2)),
                    float(coord_match.group(3))
                ]

                # Map raw values to final_coords based on directions_to_study
                # If 'x' is in directions, we take raw_values[0] and put it in final_coords[0]
                # If 'y' is in directions, we take raw_values[1] and put it in final_coords[1]
                # If 'z' is in directions, we take raw_values[2] and put it in final_coords[2]

                if 'x' in directions:
                    final_coords[0] = raw_values[0]

                if 'y' in directions:
                    final_coords[1] = raw_values[1]

                if 'z' in directions:
                    final_coords[2] = raw_values[2]

                # If a direction is NOT in directions_to_study, final_coords remains 0.0 for that index.

                parsed_k_points.append({
                    "label": key_label,
                    "coords": final_coords
                })

            else:
                raise ValueError(
                    f"Format error: Value '{value_str}' for key '{key_label}' is not a valid 3D coordinate (x, y, z). "
                    "All entries in BZ_path must be 3D."
                )

        else:
            # Line doesn't match key=value format
            print("line: " + oneLine + " is discarded.", file=sys.stderr)

    return parsed_k_points

def compute_Brillouin_zone_basis(processed_input_data):
    parsed_config=processed_input_data["parsed_config"]
    basis = parsed_config.get('lattice_basis', [])
    a0,a1,a2=basis
    a0=np.array(a0)
    a1= np.array(a1)
    a2 = np.array(a2)

    #volume, may be signed if a0,a1,a2 do not have positive orientation
    Omega=np.dot(a0,np.cross(a1,a2))

    b0=2*np.pi*np.cross(a1,a2)/Omega

    b1=2*np.pi*np.cross(a2,a0)/Omega

    b2=2*np.pi*np.cross(a0,a1)/Omega

    return b0,b1,b2


def generate_interpolation(point_start_frac, point_end_frac,BZ_basis_vectors,interpolate_point_num=15):
    # 1. Convert Fractional to Cartesian
    # We use zip to pair the coordinate component (u, v, w) with the basis vector (b0, b1, b2)
    # This automatically handles 1D, 2D, or 3D depending on the length of the inputs.
    start_cart = sum(c * b for c, b in zip(point_start_frac, BZ_basis_vectors))
    end_cart = sum(c * b for c, b in zip(point_end_frac, BZ_basis_vectors))



    # 2. Linear Interpolation
    # Create a parameter t going from 0 to 1
    t = np.linspace(0, 1, interpolate_point_num)
    # Vector from start to end
    vector_diff = end_cart - start_cart
    # Calculate path: Start + t * (End - Start)
    # np.outer allows us to multiply the shape (N,) t array by the shape (3,) vector
    # each row is an interpolated point
    interpolated_cart_coords = start_cart + np.outer(t, vector_diff)
    # 3. Calculate Distances
    # Euclidean distance of the full segment
    segment_length = np.linalg.norm(vector_diff)
    distances = t * segment_length

    return interpolated_cart_coords, distances


def interpolate_path(parsed_k_points, processed_input_data, interpolate_point_num=15):
    """
    Interpolates between consecutive k-points to create a continuous path in reciprocal space.

    Args:
        parsed_k_points: List of dicts, each containing 'label' and 'coords' for high-symmetry points.
        processed_input_data: Dictionary containing system configuration (lattice basis, dim).
        interpolate_point_num: Number of points to generate per segment.

    Returns:
        tuple: (all_coords, all_distances, high_symmetry_indices, high_symmetry_labels)
               - all_coords: (N, 3) array of Cartesian coordinates along the path.
               - all_distances: (N,) array of cumulative distances along the path.
               - high_symmetry_indices: List of indices in all_coords corresponding to the input k-points.
               - high_symmetry_labels: List of labels corresponding to high_symmetry_indices.
    """
    # 1. Get Reciprocal Lattice Basis Vectors
    b0, b1, b2 = compute_Brillouin_zone_basis(processed_input_data)
    # print(f"b0={b0}")
    # print(f"b1={b1}")
    # print(f"b2={b2}")
    dim = processed_input_data["parsed_config"]['dim']
    directions_to_study = processed_input_data["parsed_config"]['directions_to_study']  # e.g. ['x', 'y']

    # Construct BZ_basis_vectors based on directions_to_study
    # We check for x, y, z specifically to assign b0, b1, b2 respectively.
    # This handles cases like 2D systems in X-Z plane where directions=['x', 'z'].
    BZ_basis_vectors = []
    if 'x' in directions_to_study:
        BZ_basis_vectors.append(b0)
    if 'y' in directions_to_study:
        BZ_basis_vectors.append(b1)
    if 'z' in directions_to_study:
        BZ_basis_vectors.append(b2)

    # Validation checks
    if len(BZ_basis_vectors) == 0:
        raise ValueError("No valid directions found in 'directions_to_study'. Expected 'x', 'y', or 'z'.")

    if dim != len(BZ_basis_vectors):
        raise ValueError(f"Dimension mismatch: 'dim' is {dim}, but found {len(BZ_basis_vectors)} "
                         f"basis vectors based on directions_to_study: {directions_to_study}.")

    if len(parsed_k_points) < 2:
        raise ValueError("At least two k-points are required to interpolate a path.")

    all_coords = []
    all_distances = []
    high_symmetry_indices = []
    high_symmetry_labels = []

    cumulative_distance = 0.0
    current_index_count = 0

    # 2. Loop through consecutive pairs
    for i in range(len(parsed_k_points) - 1):
        start_point = parsed_k_points[i]
        end_point = parsed_k_points[i + 1]

        start_frac = start_point['coords']
        end_frac = end_point['coords']

        # Call the helper function
        # Note: generate_interpolation returns (coords, distances_from_start_of_segment)
        segment_coords, segment_distances = generate_interpolation(
            start_frac,
            end_frac,
            BZ_basis_vectors,
            interpolate_point_num
        )

        # 3. Accumulate Data
        # For the very first point of the entire path, we add everything.
        # For subsequent segments, we skip the first point to avoid duplication
        # because the end of segment i is the start of segment i+1.
        if i == 0:
            # Record the start label
            high_symmetry_indices.append(current_index_count)
            high_symmetry_labels.append(start_point['label'])

            # Add all points
            all_coords.append(segment_coords)
            # Add cumulative distance offset
            all_distances.append(segment_distances + cumulative_distance)

            current_index_count += len(segment_coords)
        else:
            # Skip the first point (it overlaps with previous segment's last point)
            all_coords.append(segment_coords[1:])

            # Add cumulative distance offset, skipping first distance
            all_distances.append(segment_distances[1:] + cumulative_distance)

            current_index_count += len(segment_coords) - 1

        # Update cumulative distance for the next segment
        # segment_distances[-1] is the length of the current segment
        cumulative_distance += segment_distances[-1]

        # Record the end label of this segment
        high_symmetry_indices.append(current_index_count - 1)
        high_symmetry_labels.append(end_point['label'])

    # 4. Concatenate arrays
    all_coords = np.vstack(all_coords)
    all_distances = np.concatenate(all_distances)

    return all_coords, all_distances, high_symmetry_indices, high_symmetry_labels

def obtain_quantum_number_k(all_coords,processed_input_data):
    """
    Calculates the projection of Brillouin zone points (p) onto the real-space lattice vectors (a_j).
    These projections k_i = (p · a_j) represent the dimensionless quantum numbers (phases)
     associated with the Periodic Boundary Conditions (PBC) along each lattice vector.
    Args:
        all_coords:   A numpy array of shape (N, 3) containing Cartesian coordinates
                    of points p in the Brillouin Zone.
        processed_input_data:  A dictionary containing the system configuration,
                              specifically the 'lattice_basis' (a0, a1, a2).

    Returns:
        A numpy array of shape (N, 3) containing the dimensionless quantum numbers k.
                       - Column 0: k_0 = p · a0
                       - Column 1: k_1 = p · a1
                       - Column 2: k_2 = p · a2

    """

    # 1. Extract real-space lattice vectors (a_j)
    parsed_config = processed_input_data["parsed_config"]
    basis = parsed_config.get('lattice_basis', [])
    a0, a1, a2 = basis
    a0 = np.array(a0)
    a1 = np.array(a1)
    a2 = np.array(a2)
    # 2. Stack vectors into a (3, 3) matrix where each row is a basis vector
    # Matrix A = [ -- a0 -- ]
    #            [ -- a1 -- ]
    #            [ -- a2 -- ]
    # Shape: (3, 3)
    real_space_basis_matrix = np.array([a0, a1, a2])
    # 3. Perform vectorized dot product to obtain quantum numbers k
    # Calculation: k = p @ A^T
    #
    # Dimensions:
    # all_coords (p): (N, 3)
    # real_space_basis_matrix.T (A^T): (3, 3) (Vectors become columns)
    #
    # Resulting Matrix (k): (N, 3)
    # Column 0: Projection of p onto a0 (p · a0)
    # Column 1: Projection of p onto a1 (p · a1)
    # Column 2: Projection of p onto a2 (p · a2)
    quantum_numbers_k = all_coords @ real_space_basis_matrix.T
    return quantum_numbers_k


def subroutine_get_interpolated_points_in_BZ_and_quantum_number_k(confFileName,interpolate_point_num=15):
    """
    Main driver function to load configuration, interpolate the path in the Brillouin Zone,
    and calculate the corresponding quantum numbers.

    This subroutine orchestrates the entire process:
    1. Resolves the data directory based on the provided configuration file path.
    2. Locates necessary configuration and data files (BZ_path.conf and preprocessed_input.pkl).
    3. Loads system parameters (dimensionality, lattice basis).
    4. Reads high-symmetry points defined in BZ_path.conf.
    5. Interpolates a path of points between these high-symmetry points.
    6. Projects these points onto real-space lattice vectors to get quantum numbers.

    Args:
        confFileName: Path to the main configuration file (used to locate the data directory).
        interpolate_point_num: Number of points to interpolate per segment between high-symmetry points.


    Returns:
        tuple: (all_coords, all_distances, high_symmetry_indices, high_symmetry_labels, quantum_numbers_k, processed_input_data)
        - all_coords: (N, 3) array of Cartesian coordinates (p) in the BZ.
        - all_distances: (N,) array of cumulative distances along the path (for plotting).
        - high_symmetry_indices: List of indices in all_coords where high-symmetry points occur.
        - high_symmetry_labels: List of string labels for the high-symmetry points.
         - quantum_numbers_k: (N, 3) array of dimensionless quantum numbers (k = p · a_j).

    """
    # 1. Resolve the directory containing data files based on the config file path
    conf_dir = get_data_directory(confFileName)

    # 2. Locate and validate input files (BZ_path.conf and preprocessed input)
    BZ_path_and_input_files=get_file_paths(conf_dir)
    validate_BZ_path_file(BZ_path_and_input_files)

    BZ_path_file_name = BZ_path_and_input_files["BZ_path_file"]
    processed_input_file_name = BZ_path_and_input_files["preprocessed_input_file"]

    # 3. Load the preprocessed system data (lattice vectors, dimensions, etc.)
    processed_input_data = parse_preprocessed_input(processed_input_file_name)

    # 4. Parse the high-symmetry points from the BZ path configuration file
    parsed_k_points = read_BZ_path_conf(BZ_path_file_name, processed_input_data)

    # 5. Generate the interpolated path in Cartesian coordinates (p)
    # This returns the coordinates, cumulative distance for plotting, and indices/labels for the ticks.
    all_coords, all_distances, high_symmetry_indices, high_symmetry_labels = interpolate_path(parsed_k_points,
                                                                                              processed_input_data,
                                                                                              interpolate_point_num)
    # 6. Calculate dimensionless quantum numbers (k)
    # Projects the Cartesian BZ points (p) onto the real-space lattice vectors (a_j).
    quantum_numbers_k = obtain_quantum_number_k(all_coords, processed_input_data)
    name=processed_input_data["name"]


    return all_coords, all_distances, high_symmetry_indices, high_symmetry_labels,quantum_numbers_k,processed_input_data,name


