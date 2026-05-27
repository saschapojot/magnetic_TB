from multiprocessing import Pool, cpu_count
import sympy as sp
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import pickle

from plot_energy_band.load_path_in_Brillouin_zone import  subroutine_get_interpolated_points_in_BZ_and_quantum_number_k
from load_Hk_parameters.load_Hk_and_hopping import subroutine_get_Hk
from name_conventions import plotting_band_data_pkl_file_name

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def Hk_symbolic_to_np(Hk, processed_input_data):
    """
     Converts a symbolic Hamiltonian matrix (SymPy) into a fast, numerical Python function (NumPy-compatible).
     This function validates that the symbolic matrix only contains momentum variables
     k0 for x, k1 for y, k2 for z based on directions_to_study.
     It ensures all physical parameters (hopping, on-site energy)
     have already been substituted.
    Args:
        Hk: The symbolic Hamiltonian matrix (sympy.Matrix).
        processed_input_data: Dictionary containing parsed configuration, specifically 'directions_to_study'.

    Returns:
        Hk_np: A lambdified function. It takes (k0, k1, ...) as arguments and returns a numpy array.
    """
    # 1. Retrieve dimensionality and directions from the configuration
    try:
        dim = processed_input_data['dim']
        directions_to_study = processed_input_data['directions_to_study']
    except KeyError:
        raise KeyError("The processed_input_data dictionary is missing 'directions_to_study' or 'dim'.")

    # 2. Identify all free symbols currently in the Hamiltonian
    Hk_free_symbols = Hk.free_symbols

    # 3. Define the standard momentum symbols expected in the Hamiltonian
    # The naming convention is fixed: k0, k1, k2 representing kx, ky, kz.
    k0 = sp.Symbol('k0', real=True)  # corresponds to x
    k1 = sp.Symbol('k1', real=True)  # corresponds to y
    k2 = sp.Symbol('k2', real=True)  # corresponds to z

    # 4. Determine the expected variables based on directions_to_study
    expected_vars = []

    # Check for presence of x, y, z in directions_to_study
    # We assume directions_to_study is a list or string containing these characters
    if 'x' in directions_to_study:
        expected_vars.append(k0)
    if 'y' in directions_to_study:
        expected_vars.append(k1)
    if 'z' in directions_to_study:
        expected_vars.append(k2)

    # Sort expected_vars to ensure consistent argument order for lambdify (k0, then k1, then k2)
    # This is crucial so that Hk_np(val1, val2) maps val1->k0 and val2->k1 correctly.
    expected_vars.sort(key=lambda x: x.name)

    if not expected_vars:
        raise ValueError(f"directions_to_study ({directions_to_study}) must contain at least one of 'x', 'y', or 'z'.")

    expected_vars_set = set(expected_vars)

    # 5. Validation: Check for un-substituted parameters
    # We subtract the allowed k-variables from the symbols found in Hk.
    # Any remaining symbols (e.g., 't', 'mu', 'alpha') indicate incomplete substitution.
    undefined_symbols = Hk_free_symbols - expected_vars_set

    # Note: It is possible that Hk does not contain a specific k if the hopping in that direction is 0,
    # so we only check if Hk contains symbols that are NOT in expected_vars.
    if len(undefined_symbols) > 0:
        raise ValueError(
            f"The symbolic Hamiltonian contains undefined free symbols: {undefined_symbols}. "
            f"Based on directions_to_study={directions_to_study}, only variables {expected_vars_set} are allowed. "
            "Please ensure all physical parameters (hopping amplitudes, onsite energies) "
            "have been substituted with numerical values."
        )

    # 6. Create the numerical function (Lambdify)
    # We convert the SymPy expression into a native Python function backed by NumPy.
    # The resulting function 'Hk_np' will accept arguments matching 'expected_vars'.
    Hk_np = sp.lambdify(expected_vars, Hk, modules='numpy')
    return Hk_np

def generate_Hk_matrix(Hk_np, quantum_numbers_k, processed_input_data):
    """
    Generates numerical Hamiltonian matrices for every k-point in the provided path.
    Args:
        Hk_np:  The lambdified (numpy-compatible) function of the Hamiltonian.
        quantum_numbers_k: A 2D numpy array where rows are k-points, each row has length 3,
                            if the dimension of lattice is 1d or 2d, the missing dims have value 0
        processed_input_data: Dictionary containing configuration (like 'dim').

    Returns:
        Hk_matrices_all: A 3D numpy array of shape (N_k_points, Matrix_Dim, Matrix_Dim).

    """
    # 1. Determine the number of k-points (rows) to calculate
    n_row, n_col = quantum_numbers_k.shape
    #
    # print(f"n_row={n_row}, n_col={n_col}")
    # 2. Retrieve dimensionality (1D, 2D, or 3D) from the config
    try:
        dim = processed_input_data["parsed_config"]['dim']
    except KeyError:
        raise KeyError("The processed_input_data dictionary is missing 'parsed_config' or 'dim'.")

    # 3. Extract the relevant spatial coordinates.
    # the first 'dim' columns (e.g., k0, k1 for 2D).
    quantum_numbers_input = quantum_numbers_k[:, 0:dim]

    # Initialize a list to store the numerical Hamiltonian matrices
    Hk_matrices_list = []

    # 4. Iterate over each k-point (each row in the input)
    for i in range(n_row):
        # Get the specific k-point coordinates for this step
        k_point = quantum_numbers_input[i, :]

        # Pass the components of the k-point as separate arguments to the lambdified function.
        # The * operator unpacks the numpy array into positional arguments (k0, k1, etc.)
        H_k_numerical = Hk_np(*k_point)

        # Ensure the output is a numpy array (lambdify sometimes returns lists/scalars depending on backend)
        H_k_numerical = np.array(H_k_numerical, dtype=complex)

        Hk_matrices_list.append(H_k_numerical)

    # 6. Convert the list of matrices into a single 3D numpy array.
    # This stacks the matrices along a new first axis.
    # Final Shape: (n_k_points, matrix_dim, matrix_dim)
    # This format is required for efficient, vectorized diagonalization later.
    Hk_matrices_all = np.array(Hk_matrices_list)

    return Hk_matrices_all

# --- Parallel Diagonalization Functions ---
def diagonalize_chunk(matrix_chunk):
    """
     Worker function: Diagonalizes a subset (chunk) of matrices.
    Args:
        matrix_chunk:  matrix_chunk: A 3D numpy array of shape (n_chunk, dim, dim).

    Returns:
        eigenvalues_sorted: 2D array (n_chunk, matrix_dim), sorted ascending.
        eigenvectors_sorted: 3D array (n_chunk, matrix_dim, matrix_dim), columns sorted matching eigenvalues.
    """
    # 1. Diagonalize
    # np.linalg.eigh usually sorts by default, but we will enforce it below to be safe.
    eigenvalues_chunk, eigenvectors_chunk = np.linalg.eigh(matrix_chunk)
    # --- Explicit Sorting (Optional but safe) ---
    # 2. Get the indices that would sort the eigenvalues along the last axis (axis 1)
    # argsort returns indices of shape (n_chunk, dim)
    sort_indices = np.argsort(eigenvalues_chunk, axis=1)
    # 3. Reorder eigenvalues
    # We use take_along_axis to apply the sort indices to the 2D array
    eigenvalues_sorted = np.take_along_axis(eigenvalues_chunk, sort_indices, axis=1)
    # 4. Reorder eigenvectors
    # Eigenvectors are columns. The array shape is (n_chunk, row, col).
    # We need to sort the columns (axis 2) based on the eigenvalue indices.
    # We must expand dimensions of sort_indices to match the eigenvector shape: (n_chunk, 1, dim)
    sort_indices_expanded = sort_indices[:, np.newaxis, :]
    eigenvectors_sorted = np.take_along_axis(eigenvectors_chunk, sort_indices_expanded, axis=2)
    return eigenvalues_sorted, eigenvectors_sorted

def diagonalize_all_Hk_matrices(Hk_matrices_all, num_processes=None):
    """
    Parallelizes the diagonalization of the Hamiltonian matrices using multiprocessing.

    Args:
        Hk_matrices_all: 3D numpy array (n_k_points, dim, dim).
        num_processes: Number of CPU cores to use. If None, uses all available cores.

    Returns:
          all_eigenvalues: 2D array (n_k_points, dim)
          all_eigenvectors: 3D array (n_k_points, dim, dim)

    """
    # num_k_points = Hk_matrices_all.shape[0]
    # 1. Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    print(f"Parallelism={num_processes}")
    # 2. Split the data into chunks
    # np.array_split divides the array into sub-arrays along axis 0.
    # It handles cases where n_k_points is not perfectly divisible by num_processes.
    chunks = np.array_split(Hk_matrices_all, num_processes, axis=0)
    # 3. Create the Pool and map the worker function
    with Pool(processes=num_processes) as pool:
        # pool.map applies 'diagonalize_chunk' to each item in 'chunks'
        # results is a list of tuples: [(evals_1, evecs_1), (evals_2, evecs_2), ...]
        results = pool.map(diagonalize_chunk, chunks)
    # 4. Reassemble the results
    # zip(*results) unzips the list of tuples into two separate tuples:
    # one containing all eigenvalue chunks, one containing all eigenvector chunks.
    eigenvalues_list, eigenvectors_list = zip(*results)
    # Concatenate the chunks back into monolithic numpy arrays
    all_eigenvalues = np.concatenate(eigenvalues_list, axis=0)
    all_eigenvectors = np.concatenate(eigenvectors_list, axis=0)
    return all_eigenvalues, all_eigenvectors

def subroutine_eigen_problem_for_energy_band_plot(confFileName,num_processes=None,interpolate_point_num=15,verbose=True):
    """
    Orchestrates the complete calculation of energy band plotting for a given system configuration.
    This subroutine acts as the main driver. It performs the following steps:
    1. Computes the symbolic Hamiltonian (Hk) using the configuration file.
    2. Loads the k-point path in the Brillouin Zone (BZ) with interpolation.
    3. Converts the symbolic Hamiltonian into a numerical function.
    4. Generates numerical matrices for every k-point in the path.
    5. Diagonalizes all matrices in parallel to obtain eigenvalues (bands) and eigenvectors.
    6. Saves all relevant plotting data to a pickle file in the same directory as the input config.

    Args:
        confFileName: Path to the configuration file of crystal (e.g., './computation_examples/hBN/hBN_primitive.conf').
        num_processes: Number of CPU cores for parallel diagonalization. If None, uses all available.
        interpolate_point_num: Number of points to interpolate between high-symmetry points in the BZ path.
        verbose: If True, prints status messages to the console.

    Returns:
        out_pickle_file_name: The absolute path string where the data dictionary was saved.

    """
    # 1. Load the symbolic Hamiltonian (Hk)
    # This reads the hopping parameters and lattice configuration to build the symbolic matrix.
    Hk = subroutine_get_Hk(confFileName,verbose)

    # 2. Generate k-points path
    # This loads the path through high-symmetry points in the Brillouin Zone
    #and make interpolation between  high-symmetry points
    # 'quantum_numbers_k' contains the actual  k0, k1, k2 coordinates (k0 for 1d; k0, k1 for 2d; k0, k1, k2 for 3d) for every point in the path.
    # 'all_distances' is used for the x-axis when plotting the bands (cumulative distance in k-space).
    all_coords, all_distances, high_symmetry_indices, high_symmetry_labels, quantum_numbers_k, processed_input_data,name = subroutine_get_interpolated_points_in_BZ_and_quantum_number_k(
        confFileName,interpolate_point_num)

    # 3. Convert Symbolic Hk to Numerical Function
    # Validates dimensions and creates a fast lambdified function for matrix generation.
    # Iterates over all k-points in 'quantum_numbers_k' and evaluates the Hamiltonian.
    # Result is a large 3D array: (Total_K_Points, Dim, Dim).
    Hk_np = Hk_symbolic_to_np(Hk, processed_input_data)

    # 4. Generate Numerical Matrices
    Hk_matrices_all = generate_Hk_matrix(Hk_np, quantum_numbers_k, processed_input_data)

    t_diag_start = datetime.now()
    all_eigenvalues, all_eigenvectors = diagonalize_all_Hk_matrices(Hk_matrices_all, num_processes)
    t_diag_end = datetime.now()
    print("diagonalization time: ", t_diag_end-t_diag_start)
    # Extract the parent directory from the configuration file path
    conf_file_path = Path(confFileName)
    directory = conf_file_path.parent
    # Construct the output pickle file path
    out_pickle_file_name = str(directory / plotting_band_data_pkl_file_name)
    # print(f"all_eigenvalues={all_eigenvalues}")
    data_to_save = {
        "name":name,
        "all_coords": all_coords,
        "all_distances": all_distances,
        "high_symmetry_indices": high_symmetry_indices,
        "high_symmetry_labels": high_symmetry_labels,
        "quantum_numbers_k": quantum_numbers_k,
        "all_eigenvalues": all_eigenvalues,
        "all_eigenvectors": all_eigenvectors
    }

    if verbose:
        print(f"Saving energy band data to: {out_pickle_file_name}")

    with open(out_pickle_file_name, 'wb') as f:
        pickle.dump(data_to_save, f)
    return out_pickle_file_name