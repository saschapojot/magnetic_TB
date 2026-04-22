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
                 repr_s_np, repr_p_np, repr_d_np, repr_f_np,spinor_mat_representation,delta_vec):
        """
        Initialize an atom with position, orbital, and representation information
        Args:
            cell: [n0, n1, n2] unit cell indices
            frac_coord: [f0, f1, f2] fractional coordinates
            position_name:  specific atom identifier for Wyckoff positions (e.g., 'C0', 'C1')
            basis: lattice basis vectors [a0, a1, a2], stored as row vectors
            origin_cart: Cartesian origin offset from cif, always [0,0,0]
            parsed_config: configuration dict containing orbital information
            repr_s_np, repr_p_np, repr_d_np, repr_f_np: representation matrices for s,p,d,f orbitals
            spinor_mat_representation: representation on spinor, time reversal considered, U for delta=1, U tilde for delta=-1
            delta_vec: indicates time reversal, a vector of 1 and -1
        """
        # Deep copy mutable inputs
        self.n0 = deepcopy(cell[0])
        self.n1 = deepcopy(cell[1])
        self.n2 = deepcopy(cell[2])
        self.position_name = position_name  # Specific identifier for Wyckoff position (e.g., 'C0')

        # example: C0_0,C_1,
        # format: [Species][Wyckoff]_[Index]
        # the first 0 refers to a unique Wyckoff position,
        # there may be multiple atoms for this Wyckoff position,
        # the second 0,1 distinguish these atoms on the same Wyckoff position
        # the value will be set later
        self.wyckoff_instance_id = None
        self.frac_coord = deepcopy(frac_coord)
        self.basis = deepcopy(basis)
        self.origin_cart = deepcopy(origin_cart)  # ← Store origin!
        self.parsed_config = deepcopy(parsed_config)
        self.spinor_mat_representation=deepcopy(spinor_mat_representation)
        self.delta_vec=deepcopy(delta_vec)

        # Calculate Cartesian coordinates using frac_to_cartesian helper
        # The basis vectors a0, a1, a2 are primitive lattice vectors expressed in
        # Cartesian coordinates using cif's origin, so the result is
        # Cartesian coordinates using cif's origin
        self.cart_coord = frac_to_cartesian(cell, frac_coord, basis, origin_cart)
        # parse Configuration to find orbitals
        found_atom = False
        self.orbitals = []  # TODO: sorting criteria
        self.atom_type = None  # To be determined from config
        wyckoff_positions = parsed_config['Wyckoff_positions']
        wyckoff_types = parsed_config['Wyckoff_position_types']
        # Lookup by position_name (e.g., 'C0')
        for pos in wyckoff_positions:
            if pos.get('position_name') == position_name:
                # Retrieve the atom type defined in the config (e.g., 'C')
                self.atom_type = pos.get('atom_type')
                # Directly access the orbital definition
                self.orbitals = deepcopy(wyckoff_types[self.position_name]['orbitals'])
                found_atom = True
                break
        if not found_atom:
            raise ValueError(f"Configuration not found for position '{position_name}'")

        self.num_orbitals = len(self.orbitals)
        # Deep copy representation matrices (all required now)
        self.repr_s_np = deepcopy(repr_s_np)
        self.repr_p_np = deepcopy(repr_p_np)
        self.repr_d_np = deepcopy(repr_d_np)
        self.repr_f_np = deepcopy(repr_f_np)
        # Pre-compute representation matrices for this atom's orbitals
        self.orbital_representations = None
        self._compute_orbital_representations()
        self.T_tilde_list = {}
        self.T_tilde_val = {}



    def _compute_orbital_representations(self):
        """
        Pre-compute orbital representation matrices for all space group operations
        Returns a list where each element is the representation matrix for one operation
        """
        num_operations = len(self.repr_s_np)
        self.orbital_representations = []
        for op_idx in range(num_operations):
            Vs = self.repr_s_np[op_idx]
            Vp = self.repr_p_np[op_idx]
            Vd = self.repr_d_np[op_idx]
            Vf = self.repr_f_np[op_idx]
            # Get submatrix for this atom's specific orbitals
            V_submatrix = orbital_to_submatrix(self.orbitals, Vs, Vp, Vd, Vf)
            self.orbital_representations.append(V_submatrix)




    def get_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix for a specific space group operation

        Args:
            operation_idx: index of the space group operation

        Returns:
            numpy array: representation matrix for this atom's orbitals
        """
        if self.orbital_representations is None:
            raise ValueError(f"Orbital representations not computed for atom {self.atom_type}")

        if operation_idx >= len(self.orbital_representations):
            raise IndexError(f"Operation index {operation_idx} out of range")

        return self.orbital_representations[operation_idx]



    def get_sympy_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix as a sympy Matrix

        Args:
            operation_idx: index of the space group operation

        Returns:
            sympy.Matrix: representation matrix for this atom's orbitals
        """
        return sp.Matrix(self.get_representation_matrix(operation_idx))


    def __str__(self):
        """String representation for print()"""
        orbitals_str = ', '.join(self.orbitals) if self.orbitals else "None"
        repr_info = f", Repr: ✓" if self.orbital_representations is not None else ""
        return (f"Atom: {self.atom_type}, "
                f"position_name='{self.position_name}', "
                f"wyckoff_instance_id='{self.wyckoff_instance_id}', "
                f"Cell: [{self.n0}, {self.n1}, {self.n2}], "
                f"Frac: {self.frac_coord}, "
                f"Cart: {self.cart_coord}, "
                f"Orbitals: [{orbitals_str}]"
                f"{repr_info}"
                )


    def __repr__(self):
        """Detailed representation for debugging"""
        orbitals_str = ', '.join(self.orbitals) if self.orbitals else "None"
        return (f"atomIndex(cell=[{self.n0}, {self.n1}, {self.n2}], "
                f"frac_coord={self.frac_coord}, "
                f"atom_type='{self.atom_type}', "
                f"position_name='{self.position_name}', "
                 f"wyckoff_instance_id='{self.wyckoff_instance_id}', "
                f"orbitals=[{orbitals_str}]), "
                f"T_tilde_val={self.T_tilde_val}"
                )

    def get_orbital_names(self):
        """Get list of orbital names for this atom"""
        return self.orbitals


    def has_orbital(self, orbital_name):
        """Check if this atom has a specific orbital"""
        # Handle both '2s' and 's' format
        orbital_type = orbital_name.lstrip('0123456789')
        return any(orb.lstrip('0123456789') == orbital_type for orb in self.orbitals)