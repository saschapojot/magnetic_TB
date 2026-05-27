"""
- Utility module for loading and working with tight-binding Hamiltonians
This module provides functions to:
- Load saved Hamiltonian data from pickle files
- Load hopping parameters from text files
- Substitute parameters into symbolic Hamiltonians

"""

from pathlib import Path
import pickle
import numpy as np
import sympy as sp
import re
import sys
from copy import deepcopy
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from name_conventions import H_pkl_file_name,hopping_parameters_template_file_name

# Exit codes for different error types
fmtErrStr = "format error: "
formatErrCode = 1        # Format/syntax errors in conf file
valueMissingCode = 2     # Required values are missing
paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist

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

def get_data_file_paths(data_dir: str) -> dict:
    """
     Construct paths to all expected data files in the directory.

    Args:
        data_dir: data directory

    Returns:
        Dictionary mapping file types to file paths (as strings)

    """
    data_path = Path(data_dir)
    return {
        'hamiltonian_pickle': str(data_path / H_pkl_file_name),
        'parameters': str(data_path / hopping_parameters_template_file_name),
    }

def validate_data_files(file_paths_dict: dict, verbose: bool = False) -> list:
    """
    Check which data files exist.

    Args:
        file_paths_dict: Dictionary mapping file types to file paths (from get_data_file_paths)
        verbose: If True, print status messages

    Returns:
        List of missing file types
    """
    missing_files = []

    if verbose:
        print("=" * 80)
        print("CHECKING FOR DATA FILES")
        print("=" * 80)

    file_descriptions = {
        'hamiltonian_pickle': "Hamiltonian data (pickle)",
        'parameters': "Hopping parameters"
    }

    for file_type, file_path_str in file_paths_dict.items():
        file_path = Path(file_path_str)
        exists = file_path.exists()

        if verbose:
            status = "✓" if exists else "✗"
            desc = file_descriptions.get(file_type, file_type)
            status_text = "" if exists else " [MISSING]"
            print(f"{status} {desc}: {file_path.name}{status_text}")

        if not exists:
            missing_files.append(file_type)

    if verbose:
        print("=" * 80)

    return missing_files

# ==============================================================================
# Define regex patterns for parsing
# ==============================================================================
# General key=value pattern
key_value_pattern = r'^([^=\s]+)\s*=\s*([^=]*)\s*$'
# Pattern for floating point numbers (including scientific notation)
float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
# Pattern for pure real numbers (float)
# Examples: 2.3, -3.5, 1e-3, -4.2e-3, 5, -10
pure_float_pattern = rf"({float_pattern})"
# Pattern for pure imaginary numbers: imag*j or imag*i
# Examples: 2.3j, -3e-5j, 2i, -4.2e-3i
# pure_imaginary_pattern = rf"({float_pattern})[ij]"
# Pattern for complex numbers: real + imag*j or real + imag*i
# Examples: 1.5+2.3j, -1.0-3e-5j, 1+2i, -3.5e2+4.2e-3i
# complex_pattern = rf"({float_pattern})\s*([\+\-])\s*({float_pattern})[ij]"
#empty value
empty_pattern = r"^\s*$"
# Pattern for orbital names (includes principal quantum number)
# Examples: 1s, 2s, 2px, 2py, 2pz, 3dxy, 3dxz, 3dyz, 3dx2-y2, 3dz2, etc.
#TODO: check f
orbital_pattern = r"[1-7](?:s|px|py|pz|dxy|dxz|dyz|dx2-y2|dz2|fxyz|fxz2|fyz2|fx3-3xy2|f3yx2-y3|fzx2-zy2|fz3)"


# Pattern for Independent real Parts of Hopping Parameters (re_T)
# Examples: re_T^{0}_{2pz_up,2pz_up}, re_T^{1}_{2s_up,2py_down}, re_T^{2}_{2py_down,2px_down}
re_T_pattern = rf"re_T\^{{(\d+)}}_{{({orbital_pattern}_(?:up|down)),({orbital_pattern}_(?:up|down))}}"

# Pattern for Independent imaginary Parts of Hopping Parameters (im_T)
# Examples: im_T^{1}_{2s_up,2py_up}, im_T^{2}_{2s_up,2px_down}, im_T^{4}_{2py_down,2px_down}
im_T_pattern = rf"im_T\^{{(\d+)}}_{{({orbital_pattern}_(?:up|down)),({orbital_pattern}_(?:up|down))}}"

# ==============================================================================
# Define helper function to clean file contents
# ==============================================================================

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


# ==============================================================================
# Helper function to parse numeric values
# ==============================================================================
def parse_numeric_value(value_str):
    """
    Parse a numeric value , pure real.
    Args:
        value_str:  String representation of the number

    Returns:
         tuple: (parsed_value, value_type) where value_type is one of:
            - 'float': for pure real numbers
            - 'empty': for empty or whitespace-only strings
            - (None, None): if parsing fails
         Examples:
          "2.3" -> (2.3, 'float')
    """
    value_str = value_str.strip()
    if re.fullmatch(empty_pattern, value_str):
        return (0, "empty")

    # Try to parse as pure real number
    # Examples: 2.3, -3.5, 1e-3, -4.2e-3
    match = re.fullmatch(pure_float_pattern, value_str)
    if match:
        return (float(match.group(1)), 'float')

    return (None, None)


def parse_hopping_parameters(hopping_parameters_file_name):
    """
    Parse hopping parameters from a text file and create SymPy symbols.
    The file should contain lines in the format:
    - re_T^{n}_{orbital1_spin,orbital2_spin} = float_value
    - im_T^{n}_{orbital1_spin,orbital2_spin} = float_value

    Args:
        hopping_parameters_file_name: Path to the text file containing the parameters.

    Returns:
        dict: A dictionary containing:
            - 'symbols': Dictionary with keys 're_T', 'im_T', each containing
                        dictionaries mapping (tree_idx, orbital1_spin, orbital2_spin) tuples to SymPy symbols
            - 'substitution_dict': Dictionary mapping SymPy symbols to their numeric values
                                  for substitution into symbolic Hamiltonians
     Raises:
        FileNotFoundError: If the hopping parameters file doesn't exist
        ValueError: If a line has invalid format or value type for the parameter

    Examples:
        Input file content:
            re_T^{1}_{2px_up,2s_up} = 0.3
            im_T^{1}_{2px_up,2s_down} = 0.5

     Returns:
            {
                'symbols': {
                    're_T': {('1', '2px_up', '2s_up'): Symbol('re_T^{1}_{2px_up,2s_up}')},
                    'im_T': {('1', '2px_up', '2s_down'): Symbol('im_T^{1}_{2px_up,2s_down}')}
                },
                'substitution_dict': {
                    Symbol('re_T^{1}_{2px_up,2s_up}'): 0.3,
                    Symbol('im_T^{1}_{2px_up,2s_down}'): 0.5
                }
            }
    """
    # Check if file exists
    param_file = Path(hopping_parameters_file_name)
    if not param_file.exists():
        raise FileNotFoundError(f"Hopping parameters file not found: {hopping_parameters_file_name}")

    # Get cleaned lines from file
    linesWithCommentsRemoved = removeCommentsAndEmptyLines(hopping_parameters_file_name)

    # Initialize storage dictionaries
    symbols = {
        're_T': {},  # SymPy symbols for real parts
        'im_T': {}  # SymPy symbols for imaginary parts
    }

    # This dictionary maps symbols to values for easy substitution into Hamiltonian
    substitution_dict = {}

    # Process each line
    for line_num, line in enumerate(linesWithCommentsRemoved, start=1):
        # Try to match key=value pattern
        kv_match = re.fullmatch(key_value_pattern, line)
        if not kv_match:
            print(f"Warning: Line {line_num} doesn't match key=value format: {line}")
            continue

        key = kv_match.group(1).strip()
        value_str = kv_match.group(2).strip()

        # ==============================================================================
        # Try to parse re_T^{n}_{orbital1_spin,orbital2_spin} = float_value
        # ==============================================================================
        re_T_match = re.fullmatch(re_T_pattern, key)
        if re_T_match:
            # Extract the captured groups
            tree_idx = re_T_match.group(1)
            orb1_spin = re_T_match.group(2)  # e.g., "2px_up"
            orb2_spin = re_T_match.group(3)  # e.g., "2s_down"

            # Parse the numeric value
            val, val_type = parse_numeric_value(value_str)

            if val_type in ['float', 'empty']:
                # Create the SymPy symbol (must match the exact string key)
                sym = sp.Symbol(key, real=True)

                # Store in our dictionaries
                symbols['re_T'][(tree_idx, orb1_spin, orb2_spin)] = sym
                substitution_dict[sym] = val
            else:
                raise ValueError(f"Invalid numeric value for {key}: {value_str}")

            continue  # Move to the next line in the file

        # ==============================================================================
        # Try to parse im_T^{n}_{orbital1_spin,orbital2_spin} = float_value
        # ==============================================================================
        im_T_match = re.fullmatch(im_T_pattern, key)
        if im_T_match:
            # Extract the captured groups
            tree_idx = im_T_match.group(1)
            orb1_spin = im_T_match.group(2)  # e.g., "2px_up"
            orb2_spin = im_T_match.group(3)  # e.g., "2s_down"

            # Parse the numeric value
            val, val_type = parse_numeric_value(value_str)

            if val_type in ['float', 'empty']:
                # Create the SymPy symbol
                sym = sp.Symbol(key, real=True)

                # Store in our dictionaries
                symbols['im_T'][(tree_idx, orb1_spin, orb2_spin)] = sym
                substitution_dict[sym] = val
            else:
                raise ValueError(f"Invalid numeric value for {key}: {value_str}")

            continue  # Move to the next line in the file

        # If we reach here, the key didn't match re_T or im_T patterns
        print(f"Warning: Unrecognized parameter key format on line {line_num}: {key}")

    return {
        'symbols': symbols,
        'substitution_dict': substitution_dict
    }


# ==============================================================================
# Hamiltonian Data Loading
# ==============================================================================
def load_hamiltonian_data(pickle_path: str, verbose: bool = True) -> dict:
    """
    Load Hamiltonian data from pickle file.
    Args:
        pickle_path:  Path to hamiltonian_data.pkl file
        verbose:  If True, print loading information

    Returns:
         Dictionary containing all Hamiltonian data:
        {
            'hamiltonian': sympy.Matrix - The symbolic Hamiltonian
            'hamiltonian_dimension': int - Total dimension
            "basis_explanation": string,  row/column names (basis vectors)
            'unit_cell_atoms': list - atomIndex objects
            'sorted_wyckoff_instance_ids': list - Ordered atom IDs
            'block_dimensions': dict - Block sizes {atom_id: num_orbitals}
            'T_tilde_blocks': dict - Individual hopping blocks
            'lattice_basis': np.ndarray - Lattice vectors
            'space_group_origin_cartesian': np.ndarray - Origin
            'config': dict - Configuration data
            'k_symbols': list - ['k0', 'k1', 'k2']
            'creation_date': str - ISO format timestamp
            'version': str - Data format version
        }

    """
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        raise FileNotFoundError(f"Hamiltonian data file not found: {pickle_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("LOADING HAMILTONIAN DATA")
        print("=" * 80)
        print(f"Loading from: {pickle_file.name}")

    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        if verbose:
            print(f"✓ Successfully loaded Hamiltonian data")
            print(f"\nData Summary:")
            print(f"  Version: {data.get('version', 'unknown')}")
            print(f"  Creation date: {data.get('creation_date', 'unknown')}")
            print(f"  Hamiltonian dimension: {data['hamiltonian_dimension']}")
            print(f"  Number of unit cell atoms: {len(data['unit_cell_atoms'])}")
            print(f"  Number of hopping blocks: {len(data['T_tilde_blocks'])}")

        return data


    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to load pickle file. File may be corrupted: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading Hamiltonian data: {e}")


def load_hamiltonian_and_hopping_from_path(conf_file_path: str, verbose: bool = True) ->  tuple[dict, dict]:
    """
    Complete pipeline to load Hamiltonian data and hopping parameters from a configuration file path.
    Args:
        conf_file_path:
        verbose:

    Returns:

    """
    # ==============================================================================
    # STEP 1: Validate input path
    # ==============================================================================
    conf_path = Path(conf_file_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {conf_file_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("LOADING HAMILTONIAN DATA")
        print("=" * 80)
        print(f"Configuration file: {conf_path}")

    # ==============================================================================
    # STEP 2: Determine data directory and construct file paths
    # ==============================================================================
    # The Hamiltonian data is stored in the same directory as the config file
    try:
        data_dir = get_data_directory(conf_file_path)
        if verbose:
            print(f"Data directory: {data_dir}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error locating data directory: {e}")

    # Get paths to all expected data files
    file_paths = get_data_file_paths(data_dir)
    if verbose:
        print(f"\nExpected data files:")
        print(f"  Hamiltonian pickle: {Path(file_paths['hamiltonian_pickle']).name}")
        print(f"  Parameters file: {Path(file_paths['parameters']).name}")

    # ==============================================================================
    # STEP 3: Validate that required files exist
    # ==============================================================================
    missing_files = validate_data_files(file_paths, verbose=verbose)
    # Check if the essential Hamiltonian pickle file is missing
    if 'hamiltonian_pickle' in missing_files:
        raise FileNotFoundError(
            f"Required Hamiltonian data file not found: {file_paths['hamiltonian_pickle']}\n"
            f"Please run the main calculation script first to generate the data."
        )
    # Check if the parameters file is missing
    if 'parameters' in missing_files:
        raise FileNotFoundError(
            f"Required parameters file not found: {file_paths['parameters']}\n"
            f"Please run the main calculation script first to generate the parameters file."
        )

    # ==============================================================================
    # STEP 4: Load Hamiltonian data from pickle file
    # ==============================================================================
    try:
        hamiltonian_data = load_hamiltonian_data(
            file_paths['hamiltonian_pickle'],
            verbose=verbose
        )
    except Exception as e:
        raise Exception(f"Failed to load Hamiltonian data: {e}")

    # ==============================================================================
    # STEP 5: Extract and validate dimensionality information
    # ==============================================================================
    # Determine system dimensionality from k_symbols
    k_symbols = hamiltonian_data.get('k_symbols', [])
    if not k_symbols:
        raise ValueError(
            f"Missing 'k_symbols' in Hamiltonian data.\n"
            f"The data file may be corrupted or from an incompatible version.\n"
            f"Please regenerate the Hamiltonian data by running the main calculation script."
        )
    # Validate dimensionality consistency
    dim = len(k_symbols)
    config_dim = hamiltonian_data.get('config', {}).get('dim', None)
    if config_dim is not None and config_dim != dim:
        raise ValueError(
            f"Dimensionality mismatch: k_symbols indicates {dim}D "
            f"but config specifies {config_dim}D"
        )
    if verbose:
        print(f"\nSystem dimensionality: {dim}D")
        print(f"k-vector components: {', '.join(k_symbols)}")

    # ==============================================================================
    # STEP 6: Load and parse hopping parameters
    # ==============================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING HOPPING PARAMETERS")
        print("=" * 80)
        print(f"Loading from: {Path(file_paths['parameters']).name}")
    try:
        hopping_params = parse_hopping_parameters(file_paths['parameters'])
        if verbose:
            print(f"✓ Successfully loaded hopping parameters")
            print(f"\nParameter Summary:")
            print(f"  re_T parameters: {len(hopping_params['symbols']['re_T'])}")
            print(f"  im_T parameters: {len(hopping_params['symbols']['im_T'])}")
            print(f"  Total symbols: {len(hopping_params['substitution_dict'])}")

    except Exception as e:
        raise Exception(f"Failed to load hopping parameters: {e}")

    # ==============================================================================
    # STEP 7: Final validation and summary
    # ==============================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING COMPLETE")
        print("=" * 80)
        print(f"✓ Hamiltonian dimension: {hamiltonian_data['hamiltonian_dimension']}")
        print(f"✓ Number of atoms: {len(hamiltonian_data['unit_cell_atoms'])}")
        print(f"✓ System dimensionality: {dim}D")
        print(f"✓ Data version: {hamiltonian_data.get('version', 'unknown')}")
        print(f"✓ Hopping parameters loaded: {len(hopping_params['substitution_dict'])} total")
        print("=" * 80 + "\n")

    return hamiltonian_data,hopping_params

# def substitute_hopping_parameters(hamiltonian_data: dict,hopping_params:dict, verbose: bool = True) -> sp.Matrix:
#     """
#     Substitute numerical values for hopping parameters into the symbolic Hamiltonian.
#     This function takes the symbolic Hamiltonian H(k) and replaces all hopping parameter
#     symbols (re_T, im_T) with their numerical values from the hopping parameters.
#     Args:
#         hamiltonian_data:
#         hopping_params:
#         verbose:
#
#     Returns:
#
#     """
#     if 'hamiltonian' not in hamiltonian_data:
#         raise KeyError("hamiltonian_data must contain 'hamiltonian' key")
#     if 'substitution_dict' not in hopping_params:
#         raise KeyError("hopping_parameters must contain 'substitution_dict' key")
#     hamiltonian = hamiltonian_data['hamiltonian']
#     substitution_dict = hopping_params['substitution_dict']
#     if not substitution_dict:
#         raise ValueError("substitution_dict is empty - no hopping parameters to substitute")
#     original_symbols = hamiltonian.free_symbols
#     # Identify which symbols are hopping parameters
#     hopping_symbols = set(substitution_dict.keys())
#     # print(f"hopping_symbols={hopping_symbols}")
#     k_symbols_list = hamiltonian_data.get('k_symbols', [])
#     k_symbols_set = set([sp.Symbol(item, real=True) for item in k_symbols_list])
#     # Find which hopping parameters actually appear in H
#     hopping_in_H = original_symbols.intersection(hopping_symbols)
#     other_symbols = original_symbols.difference(hopping_symbols).difference(k_symbols_set)
#     # print(f"other_symbols={other_symbols}")
#     # check : other_symbols should be empty
#     if other_symbols:
#         symbol_names = ', '.join(str(s) for s in sorted(other_symbols, key=str))
#         raise ValueError(
#             f"Unexpected symbols found in Hamiltonian :\n"
#             f"  {symbol_names}\n"
#         )
#     # Check that all hopping parameters are used in the Hamiltonian
#     unused_hopping_params = hopping_symbols.difference(hopping_in_H)
#     if unused_hopping_params:
#         unused_names = ', '.join(str(s) for s in sorted(unused_hopping_params, key=str))
#         raise ValueError(
#             f"The following hopping parameters are defined but not used in the Hamiltonian:\n"
#             f"  {unused_names}\n"
#             f"This may indicate a mismatch between the parameter file and the Hamiltonian structure."
#         )
#     # Check that all hopping symbols in H have defined values
#     missing_hopping_params = hopping_in_H.difference(hopping_symbols)
#     if missing_hopping_params:
#         missing_names = ', '.join(str(s) for s in sorted(missing_hopping_params, key=str))
#         raise ValueError(
#             f"The following hopping parameters appear in the Hamiltonian but have no defined values:\n"
#             f"  {missing_names}\n"
#             f"Please ensure all hopping parameters used in H(k) are defined in the parameters file."
#         )
#     # Perform the substitution directly
#     hamiltonian_with_values = hamiltonian.subs(substitution_dict)
#     # hamiltonian_with_values=deepcopy(hamiltonian)
#     # for key, value in substitution_dict.items():
#     #     hamiltonian_with_values.subs([(key,value)])
#     # hamiltonian_with_values = sp.simplify(hamiltonian_with_values)
#     # sp.pprint(hamiltonian_with_values)
#     # df=hamiltonian_with_values-hamiltonian_with_values.H
#     # sp.pprint(f"df={df}")
#     return hamiltonian_with_values

def substitute_hopping_parameters(hamiltonian_data: dict, hopping_params: dict, verbose: bool = True) -> sp.Matrix:
    """
    Substitute numerical values for hopping parameters into the symbolic Hamiltonian.
    This function takes the symbolic Hamiltonian H(k) and replaces all hopping parameter
    symbols (re_T, im_T) with their numerical values from the hopping parameters.
    """
    if 'hamiltonian' not in hamiltonian_data:
        raise KeyError("hamiltonian_data must contain 'hamiltonian' key")
    if 'substitution_dict' not in hopping_params:
        raise KeyError("hopping_parameters must contain 'substitution_dict' key")

    hamiltonian = hamiltonian_data['hamiltonian']
    substitution_dict = hopping_params['substitution_dict']

    if not substitution_dict:
        raise ValueError("substitution_dict is empty - no hopping parameters to substitute")

    original_symbols = hamiltonian.free_symbols

    # Identify which symbols are hopping parameters
    hopping_symbols = set(substitution_dict.keys())

    k_symbols_list = hamiltonian_data.get('k_symbols', [])
    k_symbols_set = set([sp.Symbol(item, real=True) for item in k_symbols_list])

    # Find which hopping parameters actually appear in H
    hopping_in_H = original_symbols.intersection(hopping_symbols)
    other_symbols = original_symbols.difference(hopping_symbols).difference(k_symbols_set)

    # check : other_symbols should be empty
    if other_symbols:
        symbol_names = ', '.join(str(s) for s in sorted(other_symbols, key=str))
        raise ValueError(
            f"Unexpected symbols found in Hamiltonian :\n"
            f"  {symbol_names}\n"
        )

    # Check that all hopping parameters are used in the Hamiltonian
    unused_hopping_params = hopping_symbols.difference(hopping_in_H)
    if unused_hopping_params:
        unused_names = ', '.join(str(s) for s in sorted(unused_hopping_params, key=str))
        raise ValueError(
            f"The following hopping parameters are defined but not used in the Hamiltonian:\n"
            f"  {unused_names}\n"
            f"This may indicate a mismatch between the parameter file and the Hamiltonian structure."
        )

    # Check that all hopping symbols in H have defined values
    missing_hopping_params = hopping_in_H.difference(hopping_symbols)
    if missing_hopping_params:
        missing_names = ', '.join(str(s) for s in sorted(missing_hopping_params, key=str))
        raise ValueError(
            f"The following hopping parameters appear in the Hamiltonian but have no defined values:\n"
            f"  {missing_names}\n"
            f"Please ensure all hopping parameters used in H(k) are defined in the parameters file."
        )

    if verbose:
        print("Substituting parameters into Hamiltonian (optimized)...")

    # ==============================================================================
    # CRITICAL SPEEDUP: Use applyfunc and xreplace instead of subs
    # ==============================================================================
    # xreplace is exact node replacement (no algebraic evaluation overhead)
    # applyfunc does it element-by-element, avoiding matrix-level overhead
    hamiltonian_with_values = hamiltonian.applyfunc(lambda expr: expr.xreplace(substitution_dict))

    return hamiltonian_with_values

def subroutine_get_Hk(confFileName,verbose=True):

    h, hop = load_hamiltonian_and_hopping_from_path(confFileName, verbose)
    Hk = substitute_hopping_parameters(h, hop, True)
    return Hk