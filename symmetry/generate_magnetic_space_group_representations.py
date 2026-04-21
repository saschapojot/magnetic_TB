import numpy as np
import sys
import json
import re
import copy
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from name_conventions import  symmetry_matrices_file_name,representations_all_file_name

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
    sys.exit(json_err_code)


# ==============================================================================
# STEP 2: Extract space group configuration data
# ==============================================================================
# Note: All operations assume primitive cell basis unless otherwise specified
try:
    # unit cell lattice basis vectors (3x3 matrix)
    # Each row is a lattice vector in Cartesian coordinates\
    lattice_basis=parsed_config['lattice_basis']
    lattice_basis=np.array(lattice_basis,dtype=float)
    conf_file_path=parsed_config["config_file_path"]
    conf_file_dir=Path(conf_file_path).parent
    symmetry_matrices_file_name_path=str(conf_file_dir/symmetry_matrices_file_name)
    # Load the pickle file
    with open(symmetry_matrices_file_name_path, 'rb') as f:
        symmetry_matrices = pickle.load(f)


except FileNotFoundError:
    print(f"Error: Symmetry matrices file not found at {symmetry_matrices_file_name_path}", file=sys.stderr)
    sys.exit(file_err_code)
except KeyError as e:
    print(f"Error: Missing required key in configuration: {e}", file=sys.stderr)
    sys.exit(key_err_code)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(val_err_code)

###computing spinor representations

# Define Pauli matrices globally
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I_2x2 = np.eye(2, dtype=complex)

def O3_to_spinor(R, tol=1e-7):
    """
    Computes the spinor transformation matrix for a given O(3) matrix R.

    Args:
        R (numpy.ndarray): 3x3 orthogonal matrix in O(3).
        tol (float): Tolerance for floating point comparisons.

    Returns:
        (numpy.ndarray): U
    """
    R = np.array(R, dtype=float)

    # Step 1: Get pure rotation matrix B
    det_R = np.linalg.det(R)
    B = det_R * R

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


def U_with_delta(R,delta,tol=1e-7):
    """

    Args:
        R: O(3) matrix
        delta: 1 means no time reversal, -1 means having time reversal

    Returns: U or U_tilde

    """
    delta=int(delta)
    U=O3_to_spinor(R,tol)
    if delta==1:
        return U

    elif delta==-1:
        U_tilde=-1j*sigma_y@np.conj(U)
        return U_tilde
    else:
        raise ValueError(f"Invalid value for delta: {delta}. Expected 1 or -1.")



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
        R_trans = np.array(dict_item["mat"], dtype=float)
        delta = dict_item["delta"]
        R = R_trans[:, 0:3]
        trans=R_trans[:, 3]
        # Transform rotation/reflection part
        magnetic_space_group_matrices_spatial_cartesian[j, :, 0:3]=basis_T@R@basis_T_inv
        # Transform translation part
        magnetic_space_group_matrices_spatial_cartesian[j, :, 3] =basis_T@trans
        delta_vec.append(delta)
    spinor_mat_representation=[]
    #compute spinor_mats
    for j in range(0,num_operators):
        R_cart = magnetic_space_group_matrices_spatial_cartesian[j, :, 0:3]


        delta=delta_vec[j]
        spinor_mat=U_with_delta(R_cart,delta,tol)
        spinor_mat_representation.append(spinor_mat)

    spinor_mat_representation = np.array(spinor_mat_representation)
    delta_vec = np.array(delta_vec)  # Convert to numpy array for consistency
    return magnetic_space_group_matrices_spatial_cartesian,spinor_mat_representation,delta_vec



# ==============================================================================
#  Define orbital representation functions
# ==============================================================================
def magnetic_space_group_representation_D_orbitals(R):
    """
    Compute how a symmetry operation acts on d orbitals

    Original function: GetSymD(R) in cd/SymGroup.py

    The d orbitals transform as quadratic functions of coordinates:
    d_xy, d_yz, d_xz, d_(x²-y²), d_(3z²-r²)

    This function computes the 5x5 representation matrix showing how
    the rotation R transforms the d orbital basis.

    :param R: Linear part of space group operation (3x3 rotation matrix) in Cartesian basis
    :return: Representation matrix (5x5) for d orbitals
    """
    [[R_11, R_12, R_13], [R_21, R_22, R_23], [R_31, R_32, R_33]] = R
    RD = np.zeros((5, 5))
    sr3 = np.sqrt(3)

    # Row 0: d_xy orbital transformation
    RD[0, 0] = R_11*R_22 + R_12*R_21
    RD[0, 1] = R_21*R_32 + R_22*R_31
    RD[0, 2] = R_11*R_32 + R_12*R_31
    RD[0, 3] = 2*R_11*R_12 + R_31*R_32
    RD[0, 4] = sr3*R_31*R_32

    # Row 1: d_yz orbital transformation
    RD[1, 0] = R_12*R_23 + R_13*R_22
    RD[1, 1] = R_22*R_33 + R_23*R_32
    RD[1, 2] = R_12*R_33 + R_13*R_32
    RD[1, 3] = 2*R_12*R_13 + R_32*R_33
    RD[1, 4] = sr3*R_32*R_33

    # Row 2: d_zx orbital transformation
    RD[2, 0] = R_11*R_23 + R_13*R_21
    RD[2, 1] = R_21*R_33 + R_23*R_31
    RD[2, 2] = R_11*R_33 + R_13*R_31
    RD[2, 3] = 2*R_11*R_13 + R_31*R_33
    RD[2, 4] = sr3*R_31*R_33

    # Row 3: d_(x²-y²) orbital transformation
    RD[3, 0] = R_11*R_21 - R_12*R_22
    RD[3, 1] = R_21*R_31 - R_22*R_32
    RD[3, 2] = R_11*R_31 - R_12*R_32
    RD[3, 3] = (R_11**2 - R_12**2) + 1/2*(R_31**2 - R_32**2)
    RD[3, 4] = sr3/2*(R_31**2 - R_32**2)

    # Row 4: d_(3z²-r²) orbital transformation
    RD[4, 0] = 1/sr3*(2*R_13*R_23 - R_11*R_21 - R_12*R_22)
    RD[4, 1] = 1/sr3*(2*R_23*R_33 - R_21*R_31 - R_22*R_32)
    RD[4, 2] = 1/sr3*(2*R_13*R_33 - R_11*R_31 - R_12*R_32)
    RD[4, 3] = 1/sr3*(2*R_13**2 - R_11**2 - R_12**2) + 1/sr3/2*(2*R_33**2 - R_31**2 - R_32**2)
    RD[4, 4] = 1/2*(2*R_33**2 - R_31**2 - R_32**2)

    return RD.T

def magnetic_space_group_representation_F_orbitals(R):
    """
    Compute how a symmetry operation acts on f orbitals

    Original function: GetSymF(R) in cd/SymGroup.py

    The f orbitals transform as cubic functions of coordinates:
    fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)

    This function computes the 7x7 representation matrix showing how
    the rotation R transforms the f orbital basis.

    :param R: Linear part of space group operation (3x3 rotation matrix) in Cartesian basis
    :return: Representation matrix (7x7) for f orbitals
    """
    sr3 = np.sqrt(3)
    sr5 = np.sqrt(5)
    sr15 = np.sqrt(15)

    # Define cubic monomials: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    x1x2x3 = np.array([
        [1, 1, 1],  # x³
        [2, 2, 2],  # y³
        [3, 3, 3],  # z³
        [1, 1, 2],  # x²y
        [1, 2, 2],  # xy²
        [1, 1, 3],  # x²z
        [1, 3, 3],  # xz²
        [2, 2, 3],  # y²z
        [2, 3, 3],  # yz²
        [1, 2, 3]   # xyz
    ], int)

    # Compute how rotation R acts on cubic monomials
    # Rx1x2x3[i,j] = coefficient of monomial j in transformed monomial i
    Rx1x2x3 = np.zeros((10, 10))
    for i in range(10):
        n1, n2, n3 = x1x2x3[i]
        # Transform each cubic monomial by applying R to each factor
        Rx1x2x3[i, 0] = R[1-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1]  # x³
        Rx1x2x3[i, 1] = R[2-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1]  # y³
        Rx1x2x3[i, 2] = R[3-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1]  # z³
        # x²y (sum of all permutations)
        Rx1x2x3[i, 3] = (R[1-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1] +
                         R[1-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1])
        # xy² (sum of all permutations)
        Rx1x2x3[i, 4] = (R[1-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1] +
                         R[2-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1])
        # x²z (sum of all permutations)
        Rx1x2x3[i, 5] = (R[1-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1] +
                         R[1-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1])
        # xz² (sum of all permutations)
        Rx1x2x3[i, 6] = (R[1-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1] +
                         R[3-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1])
        # y²z (sum of all permutations)
        Rx1x2x3[i, 7] = (R[2-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1] +
                         R[2-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1])
        # yz² (sum of all permutations)
        Rx1x2x3[i, 8] = (R[2-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1] +
                         R[3-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1])
        # xyz (sum of all 6 permutations)
        Rx1x2x3[i, 9] = (R[1-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1] +
                         R[1-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1] +
                         R[2-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1])

    # Matrix to express f orbitals as linear combinations of cubic monomials
    # Rows: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    # Columns: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    F = np.array([
        [       0,        0,   1/sr15,        0,        0, -3/2/sr15,        0, -3/2/sr15,        0,        0],  # fz³
        [-1/2/sr5,        0,        0,        0, -1/2/sr5,        0,    2/sr5,        0,        0,        0],  # fxz²
        [       0, -1/2/sr5,        0, -1/2/sr5,        0,        0,        0,        0,    2/sr5,        0],  # fyz²
        [       0,        0,        0,        0,        0,        0,        0,        0,        0,        1],  # fxyz
        [       0,        0,        0,        0,        0,      1/2,        0,     -1/2,        0,        0],  # fz(x²-y²)
        [ 1/2/sr3,        0,        0,        0,   -sr3/2,        0,        0,        0,        0,        0],  # fx(x²-3y²)
        [       0, -1/2/sr3,        0,    sr3/2,        0,        0,        0,        0,        0,        0]   # fy(3x²-y²)
    ])

    # Transform f orbitals: FR = F @ Rx1x2x3
    FR = F @ Rx1x2x3  # Shape: (7, 10)

    # Matrix to convert back from cubic monomials to f orbitals
    # Rows: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    # Columns: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    CF = np.array([
        [     0,      0,   sr15,      0,      0,      0,      0,      0,      0,      0],  # fz³
        [     0,      0,      0,      0,      0,      0,  sr5/2,      0,      0,      0],  # fxz²
        [     0,      0,      0,      0,      0,      0,      0,      0,  sr5/2,      0],  # fyz²
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      1],  # fxyz
        [     0,      0,      3,      0,      0,      2,      0,      0,      0,      0],  # fz(x²-y²)
        [ 2*sr3,      0,      0,      0,      0,      0,  sr3/2,      0,      0,      0],  # fx(x²-3y²)
        [     0, -2*sr3,      0,      0,      0,      0,      0,      0, -sr3/2,      0]   # fy(3x²-y²)
    ])

    # Final representation matrix for f orbitals
    RF = FR @ CF.T
    return RF.T



def magnetic_space_group_representation_orbitals_all(magnetic_space_group_matrices_cartesian):
    """
    Compute magnetic space group representations for all atomic orbital types
    For each symmetry operation in the space group, compute how it transforms:
    - s orbitals (scalar, trivial representation)
    - p orbitals (3D vector: px, py, pz)
    - d orbitals (5D: dxy, dyz, dxz, d(x²-y²), d(3z²-r²))
    - f orbitals (7D: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²))

    Args:
        magnetic_space_group_matrices_cartesian: magnetic Space group matrices (affine) under Cartesian basis

    Returns: List of representations [repr_s, repr_p, repr_d, repr_f]

    """
    num_matrices, _, _ = magnetic_space_group_matrices_cartesian.shape
    # s orbitals: spherically symmetric, trivial representation (all 1's)
    repr_s = np.ones((num_matrices, 1, 1))

    # p orbitals: transform as vectors (px, py, pz)
    # Use the rotation part of the space group matrices
    repr_p = copy.deepcopy(magnetic_space_group_matrices_cartesian[:, :3, :3])

    # d orbitals: 5x5 representation
    # Basis: dxy, dyz, dxz, d(x²-y²), d(3z²-r²)
    repr_d = np.zeros((num_matrices, 5, 5))
    for i in range(num_matrices):
        R = magnetic_space_group_matrices_cartesian[i, :3, :3]
        repr_d[i] = magnetic_space_group_representation_D_orbitals(R)

    # f orbitals: 7x7 representation
    # Basis: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    #TODO: check order of basis
    repr_f = np.zeros((num_matrices, 7, 7))
    for i in range(num_matrices):
        R = magnetic_space_group_matrices_cartesian[i, :3, :3]
        repr_f[i] = magnetic_space_group_representation_F_orbitals(R)

    repr_s_p_d_f = [repr_s, repr_p, repr_d, repr_f]
    return repr_s_p_d_f


def subroutine_generate_all_representations(symmetry_matrices,lattice_basis,tol=1e-7):
    """

    Args:
        symmetry_matrices:
        lattice_basis:

    Returns:

    """
    magnetic_space_group_matrices_spatial_cartesian, spinor_mat_representation, delta_vec=magnetic_space_group_to_cartesian_basis_and_spinor(symmetry_matrices,lattice_basis,tol)
    repr_s_p_d_f=magnetic_space_group_representation_orbitals_all(magnetic_space_group_matrices_spatial_cartesian)

    # repr_s, repr_p, repr_d, repr_f = repr_s_p_d_f

    # Create output dictionary with all computed representations as NumPy arrays
    magnetic_space_group_representations = {
        "magnetic_space_group_matrices_spatial_cartesian": magnetic_space_group_matrices_spatial_cartesian,
        "spinor_mat_representation": spinor_mat_representation,
        "delta_vec": delta_vec,
        "repr_s_p_d_f": [
            repr_s_p_d_f[0],  # s orbital representation
            repr_s_p_d_f[1],  # p orbital representation
            repr_s_p_d_f[2],  # d orbital representation
            repr_s_p_d_f[3]  # f orbital representation
        ]
    }



    return magnetic_space_group_representations

tol=1e-7
representations = subroutine_generate_all_representations(symmetry_matrices,lattice_basis,tol)
# Save the representations dictionary to a pickle file
representations_all_file_name_path = Path(conf_file_dir) / representations_all_file_name
with open(representations_all_file_name_path, 'wb') as f:
    pickle.dump(representations, f)