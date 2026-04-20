import numpy as np
import sys
import json
import re
import copy
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from name_conventions import  symmetry_matrices_file_name

# ==============================================================================
# Magnetic space group representation computation script
# ==============================================================================
# This script computes magnetic space group representations for atomic orbitals, and for spins in sigma_z basis

#for orbital part, the magnetic space groups are in  Cartesian basis (x, y, z coordinates)
# It also computes how symmetry operations act on atomic orbitals (s, p, d, f)
#for the spin part, it computes the unitary representation, under the sign of delta
# Exit codes for different error conditions
json_err_code = 4   # JSON parsing error
key_err_code = 5    # Required key missing from configuration
val_err_code = 6    # Invalid value in configuration
file_err_code = 7   # File not found or IO error

# ==============================================================================
# STEP 1: Read and parse JSON input from stdin
# ==============================================================================

try:
    config_json = sys.stdin.read()
    parsed_config = json.loads(config_json)

except json.JSONDecodeError as e:
    print(f"Error parsing JSON input: {e}", file=sys.stderr)
    exit(json_err_code)


# ==============================================================================
# STEP 2: Extract space group configuration data
# ==============================================================================
# Note: All operations assume primitive cell basis unless otherwise specified
try:
    # unit cell lattice basis vectors (3x3 matrix)
    # Each row is a lattice vector in Cartesian coordinates\
    lattice_basis=parsed_config['lattice_basis']
    lattice_basis=np.array(lattice_basis)
    conf_file_path=parsed_config["config_file_path"]
    conf_file_dir=Path(conf_file_path).parent
    symmetry_matrices_file_name_path=str(conf_file_dir/symmetry_matrices_file_name)
    # Load the pickle file
    with open(symmetry_matrices_file_name_path, 'rb') as f:
        symmetry_matrices = pickle.load(f)


except FileNotFoundError:
    print(f"Error: Symmetry matrices file not found at {symmetry_matrices_file_name_path}", file=sys.stderr)
    exit(file_err_code)
except KeyError as e:
    print(f"Error: Missing required key in configuration: {e}", file=sys.stderr)
    exit(key_err_code)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    exit(val_err_code)

###computing spinor representations

# Define Pauli matrices globally
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I_2x2 = np.eye(2, dtype=complex)

def O3_to_spinor(A, tol=1e-7):
    """
    Computes the spinor transformation matrix for a given O(3) matrix A.

    Args:
        A (numpy.ndarray): 3x3 orthogonal matrix in O(3).
        tol (float): Tolerance for floating point comparisons.

    Returns:
        (numpy.ndarray): U
    """
    A = np.array(A, dtype=float)

    # Step 1: Get pure rotation matrix B
    det_A = np.linalg.det(A)
    B = det_A * A

    # Step 2: Compute rotation angle theta
    # Clip trace to [-1, 3] to avoid NaN in arccos due to floating point inaccuracies
    tr_B = np.clip(np.trace(B), -1.0, 3.0)
    theta = np.arccos((tr_B - 1.0) / 2.0)

    # Step 3: Compute rotation axis n = [n0, n1, n2]
    sin_theta = np.sin(theta)
    if abs(sin_theta) > tol:
        n0 = (B[2, 1] - B[1, 2]) / (2 * sin_theta)
        n1 = (B[0, 2] - B[2, 0]) / (2 * sin_theta)
        n2 = (B[1, 0] - B[0, 1]) / (2 * sin_theta)
    else:
        if abs(theta - np.pi) <= tol:
            # theta = pi case
            abs_n0 = np.sqrt(max(0, (1 + B[0, 0]) / 2))
            abs_n1 = np.sqrt(max(0, (1 + B[1, 1]) / 2))
            abs_n2 = np.sqrt(max(0, (1 + B[2, 2]) / 2))

            if np.abs(B[0, 0] + 1.0) > tol:  # Case 1: B00 != -1 (i.e., n0 != 0)
                n0 = abs_n0
                n1 = np.sign(B[0, 1]) * abs_n1
                n2 = np.sign(B[0, 2]) * abs_n2
            elif np.abs(B[0, 0] + 1.0) <= tol and np.abs(B[1, 1] + 1.0) > tol:
                n0 = 0.0
                n1 = abs_n1
                n2 = np.sign(B[1, 2]) * abs_n2
            else:
                n0 = 0.0
                n1 = 0.0
                n2 = abs_n2
        else:
            # theta = 0 case, axis is arbitrary
            n0, n1, n2 = 0.0, 0.0, 1.0

    # Normalize n to correct any floating point drift
    n = np.array([n0, n1, n2])
    n_norm = np.linalg.norm(n)
    if n_norm > 0:
        n /= n_norm
    n0, n1, n2 = n

    # Step 4: Construct the SU(2) matrix U using global Pauli matrices
    n_dot_sigma = n0 * sigma_x + n1 * sigma_y + n2 * sigma_z
    U = np.cos(theta / 2) * I_2x2 - 1j * np.sin(theta / 2) * n_dot_sigma
    return U


def U_with_delta(A,delta,tol=1e-7):
    """

    Args:
        A: O(3) matrix
        delta: 1 means no time reversal, -1 means having time reversal

    Returns: U or U_tilde

    """
    delta=int(delta)
    U=O3_to_spinor(A,tol)
    if delta==1:
        return U

    elif delta==-1:
        U_tilde=-1j*sigma_y@np.conj(U)
        return U_tilde
    else:
        raise ValueError(f"Invalid value for delta: {delta}. Expected 1 or -1.")


num=len(symmetry_matrices)
print(f"num={num}")
n=15
dict_item=symmetry_matrices[n]
A_trans=dict_item["mat"]
delta=dict_item["delta"]
A=A_trans[:,:3]
print(f"A={A}, delta={delta}")
spinor_repr_mat=U_with_delta(A,delta)
print(f"spinor_repr_mat={spinor_repr_mat}")

# ==============================================================================
# Define coordinate transformation functions
# ==============================================================================

def magnetic_space_group_to_cartesian_basis_and_spinor(symmetry_matrices,lattice_basis,tol=1e-7):
    """
    in .cif file, the symmetry_matrices are compatible with the lattice_basis
    Args:
        symmetry_matrices: a list of dict, each dict contains the spatial part and delta
        lattice_basis:

    Returns:

    """
    basis_T = lattice_basis.T  # Transpose for column-vector representation
    basis_T_inv = np.linalg.inv(basis_T)

    num_operators = len(symmetry_matrices)
    magnetic_space_group_matrices_spatial_cartesian = np.zeros((num_operators, 3, 4), dtype=float)
    delta_vec=[]
    for j,dict_item in enumerate(symmetry_matrices):
        A_trans = dict_item["mat"]
        delta = dict_item["delta"]
        A = A_trans[:, 0:3]
        trans=A_trans[:, 3]
        # Transform rotation/reflection part
        magnetic_space_group_matrices_spatial_cartesian[j, :, 0:3]=basis_T@A@basis_T_inv
        # Transform translation part
        magnetic_space_group_matrices_spatial_cartesian[j, :, 3] =basis_T@trans
        delta_vec.append(delta)
    spinor_mat_representation=[]
    #compute spinor_mats
    for j in range(0,num_operators):
        A_cart = magnetic_space_group_matrices_spatial_cartesian[j, :, 0:3]


        delta=delta_vec[j]
        spinor_mat=U_with_delta(A_cart,delta,tol)
        spinor_mat_representation.append(spinor_mat)

    spinor_mat_representation=np.array(spinor_mat_representation)
    return magnetic_space_group_matrices_spatial_cartesian,spinor_mat_representation,delta_vec






