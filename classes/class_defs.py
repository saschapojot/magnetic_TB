import numpy as np
from copy import deepcopy
from scipy.linalg import block_diag
import sympy as sp
import re
from datetime import datetime
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


from name_conventions import  full_orbitals

def orbital_to_submatrix(orbitals, Vs, Vp, Vd, Vf):
    """
    Extract submatrix from full orbital representation for specific orbitals

    Args:
        orbitals: List of orbital names (e.g., ['2s', '2px', '2py', '2pz'])
        Vs, Vp, Vd, Vf: Representation matrices for s, p, d, f orbitals

    Returns:
        numpy array: Submatrix for the specified orbitals
    """
    # Remove leading numbers from orbitals (e.g., '2s' -> 's', '2pz' -> 'pz')
    orbital_types = []
    for orb in orbitals:
        # Remove all leading digits
        orbital_type = orb.lstrip('0123456789')
        orbital_types.append(orbital_type)
    # Sort orbitals by their position in full_orbitals
    sorted_orbital_types = sorted(orbital_types, key=lambda orb: full_orbitals.index(orb))
    # Get the indices in full_orbitals
    orbital_indices = [full_orbitals.index(orb) for orb in sorted_orbital_types]
    # Build full representation matrix
    hopping_matrix_full = block_diag(Vs, Vp, Vd, Vf)
    # Extract submatrix for the specific orbitals
    V_submatrix = hopping_matrix_full[np.ix_(orbital_indices, orbital_indices)]
    return V_submatrix


def frac_to_cartesian(cell, frac_coord, basis,origin_cart):
    """
    Convert fractional coordinates to Cartesian coordinates.
    The transformation is:
        r_cart = (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2 + origin
    where:
        - [n0, n1, n2] are unit cell indices
        - [f0, f1, f2] are fractional coordinates within the cell
        - [a0, a1, a2] are lattice basis vectors
        - origin is the coordinate system origin (e.g., Bilbao origin)
    Args:
        cell: [n0, n1, n2] unit cell indices
        frac_coord:  [f0, f1, f2] fractional coordinates within the unit cell
        basis: lattice basis vectors, 3×3 array where each row is a basis vector
        origin_cart: Origin offset in Cartesian coordinates, For Bilbao convention, use space_group_origin_cartesian

    Returns:
        numpy array: Cartesian coordinates (3D vector)
    """
    n0, n1, n2 = cell
    f0, f1, f2 = frac_coord
    a0, a1, a2 = basis
    # Compute Cartesian position from fractional coordinates
    r_cart = (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2+ np.array(origin_cart)
    return r_cart

class atomIndex:
    def __init__(self, cell, frac_coord,position_name, basis,origin_cart, parsed_config,
                 repr_s_np, repr_p_np, repr_d_np, repr_f_np):