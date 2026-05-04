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
        return (f"Atom: {self.wyckoff_instance_id}, "
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





# ==============================================================================
# hopping class
# ==============================================================================
class hopping:
    """
    Represents a single hopping relation from a neighbor atom to a center atom.
    The hopping direction is: to_atom (center) ← from_atom (neighbor)
    The hopping is defined by a magnetic space group spatial operation ⊗ spinor part that transforms a seed hopping.
    This hopping is obtained from seed hopping by transformation:
    r' = R @ r + t + n₀·a₀ + n₁·a₁ + n₂·a₂, δ=±1
    where R is rotation, t is translation, and n_vec = [n₀, n₁, n₂] is the lattice shift,
    and r is the position vector from seed hopping's from_atom (neighbor) to to_atom (center),
    δ=±1
    """

    def __init__(self, to_atom, from_atom,
                 operation_idx,
                 #rotation_matrix, translation_vector,
                 n_vec,
                 #spinor_mat,delta,
                 is_seed):
        """
         Initialize a hopping relation: to_atom (center) ← from_atom (neighbor).
         This hopping is generated by applying a magnetic space group operation to a seed hopping.
         The transformation maps the seed neighbor position to this hopping's neighbor position.
        Args:
            to_atom:  atomIndex object for the center atom (hopping destination)
            from_atom:  atomIndex object for the neighbor atom (hopping source)
            operation_idx: Index of the magnetic space group operation that generates this hopping
                          from the seed hopping in the equivalence class
            rotation_matrix:  3×3 rotation/reflection matrix R (in Cartesian coordinates, cif origin)
            translation_vector:  3D translation vector t from the cif magnetic space group operation
                              (in Cartesian coordinates, cif origin)
            n_vec:   Array [n₀, n₁, n₂] containing integer coefficients for lattice translation
                    This is the additional lattice shift that is not given by cif data
                    The full spatial transformation is: r' = R @ r + t + n₀·a₀ + n₁·a₁ + n₂·a₂
                    Note that cif only gives R and t
            spinor_mat: spinor transformation matrix
            delta: ±1, 1 means no time reversal, -1 means time reversal
            is_seed:
        """

        self.to_atom = deepcopy(to_atom)  # Deep copy of center atom (destination)
        self.from_atom = deepcopy(from_atom)  # Deep copy of neighbor atom (source)
        self.operation_idx = operation_idx  # Which magnetic space group operation transforms parent hopping to this hopping
       # self.rotation_matrix = deepcopy(rotation_matrix)  # Deep copy of 3×3 cif rotation  matrix R
       #  self.translation_vector = deepcopy(translation_vector)  # Deep copy of 3D cif translation t
        self.n_vec = np.array(n_vec)  # Lattice translation coefficients [n₀, n₁, n₂]
        # Additional lattice shift not given by cif data
        # Computed to preserve center atom invariance
        # self.spinor_mat=deepcopy(spinor_mat)
        # self.delta=delta
        self.is_seed = is_seed  # Boolean: True if this is the seed hopping, False if derived from seed (parent)
        self.distance = None  # Euclidean distance between center (to_atom) and neighbor (from_atom)
        self.T = None  # Hopping matrix between orbital basis (sympy Matrix, to be computed)
        # Represents the tight-binding hopping matrix: center orbitals with spin ← neighbor orbitals with spin
        self.line_type = None  # for visualization
        self.T_reconstructed = None  # reconstructed by constraints
        self.T_reconstructed_swap = None  # swapping constraints, computed from T_reconstructed

    def conjugate(self):
        """
        Return the conjugate (reverse) hopping direction.
         For this hopping: center ← neighbor, the conjugate is: neighbor ← center.
         This is used to enforce Hermiticity constraints in tight-binding models:
         T(neighbor ← center) = T(center ← neighbor)†
        :return: list: [from_atom, to_atom] with swapped order (deep copied)
                 Represents the reverse hopping: neighbor ← center
        """
        return [deepcopy(self.from_atom), deepcopy(self.to_atom)]


    def compute_distance(self):
        """
         Compute the Euclidean distance from the neighbor atom to the center atom.
         This distance is calculated in Cartesian coordinates using cif origin.
         All hoppings in the same equivalence class should have the same distance
         (up to numerical precision), as they are related by symmetry operations.
        Adds member variable self.distance: L2 norm of the position difference vector (center - neighbor)
        """
        pos_to = self.to_atom.cart_coord  # Cartesian position of center atom
        pos_from = self.from_atom.cart_coord # Cartesian position of neighbor atom

        # Real space position difference vector (center - neighbor)
        delta_pos = pos_to - pos_from
        # Compute Euclidean distance (L2 norm)
        self.distance = np.linalg.norm(delta_pos, ord=2)


    def __repr__(self):
        """
        String representation for debugging and display.

        :return: str: Compact representation showing: center_type[n0,n1,n2] ← neighbor_type[m0,m1,m2],
                 operation index, distance, and seed status
        """
        seed_marker = " [SEED]" if self.is_seed else ""
        distance_str = f"{self.distance:.4f}" if self.distance is not None else "None"

        # Format cell indices for to_atom and from_atom
        to_cell = f"[{self.to_atom.n0},{self.to_atom.n1},{self.to_atom.n2}]"
        from_cell = f"[{self.from_atom.n0},{self.from_atom.n1},{self.from_atom.n2}]"

        return (f"hopping({self.to_atom.atom_type}{to_cell} ← {self.from_atom.atom_type}{from_cell}, "
                f"op={self.operation_idx}, "
                f"distance={distance_str}"
                f"{seed_marker})")



# ==============================================================================
# vertex class
# ==============================================================================

class vertex():
    """
    Represents a node in the symmetry constraint tree for tight-binding hopping matrices.
    Each vertex contains a hopping object, the hopping object contains hopping matrix of to_atom (center) ← from_atom (neighbor)
    The tree structure represents how parent hopping generates this hopping by magnetic space group operations or Hermiticity constraints.

    Tree Structure:
        - Root vertex: Corresponds to the seed hopping (identity operation)
        - Child vertices: Hoppings derived from parent through symmetry operations or Hermiticity
        - Constraint types: "linear" (from magnetic space group) or "hermitian" (from H† = H)

    The tree is used to:
    1. Express derived hopping matrices in terms of independent matrices (in root)
    2. Enforce symmetry constraints automatically
    3. Reduce the number of independent tight-binding parameters

    CRITICAL: Tree Structure Uses References (Pointers)
    ================================================
    The parent-child relationships are implemented using REFERENCES (C++ sense) / POINTERS (C sense):
    - self.parent stores a REFERENCE to the parent vertex object (not a copy)
    - self.children stores a list of REFERENCES to child vertex objects (not copies)

    This means:
    - Multiple vertices can reference the same parent object
    - Modifying a parent's hopping matrix T affects all children's constraint calculations
    - The tree forms a true graph structure in memory with shared nodes
    - Deleting a vertex requires careful handling to avoid dangling references


    Memory Diagram Example:
    ----------------------
    Root Vertex (id=0x1000) ──┬──> Child 0, linear (address=0x2000, parent address=0x1000)
                              ├──> Child 1, linear (address=0x3000, parent address=0x1000)
                              └──> Child 2, hermitian (address=0x4000, parent address=0x1000)
    All three children have parent=0x1000 (same memory address)
    Root's self.children = [0x2000, 0x3000, 0x4000] (references, not copies)
    """

    def __init__(self, hopping, type, identity_idx, parent=None):
        """
        Initialize a vertex in the tree.
        Args:
            hopping:  hopping object representing the tight-binding term: center ← neighbor
                      Contains the hopping matrix T between orbital basis, T's row represents: center atom orbitals with spin,
                      T's column represents: neighbor atom orbitals with spin
                      one element in T is the hopping coefficient from one orbital with spin in neighbor atom to
                      one orbital with spin in center atom
            type:      Constraint type that shows how this vertex is derived from its parent
                            - "linear": Derived from parent via magnetic space group symmetry operation
                            - "hermitian": Derived from parent via Hermiticity constraint
                            - None: It is root vertex

            identity_idx:  Index of the identity operation in magnetic space group
                           Used to identify root vertices (hopping.operation_idx == identity_idx)
            parent: REFERENCE to parent vertex object (default: None for root)
                    NOT deep copied - this is a reference (C++ sense) / pointer (C sense)

                    Why parent is a reference:
                    -------------------------
                    1. Upward Traversal: Allows child → parent → root navigation
                    2. Constraint Access: Child can read parent's hopping matrix T
                    3. Shared Parent: Multiple children reference same parent object
                    IMPORTANT: parent=None only for root vertices
                               parent≠None for all derived vertices (children)
        """
        self.hopping = deepcopy(hopping)  # Deep copy of hopping object containing:
                                          # - to_atom (center), from_atom (neighbor)
                                          # - is_seed, operation_idx
                                          # - rotation_matrix R, translation_vector t, n_vec, spinor_mat,delta, is_seed
                                          # - distance, T (hopping matrix)

        self.type = type  # Constraint type: None (root), "linear" (symmetry), or "hermitian"
                          # String is immutable, safe to assign directly

        self.is_root = (hopping.operation_idx == identity_idx)  # Boolean flag identifying root vertex
                                                                # Root vertex contains identity operation
                                                                # Starting vertex of hopping matrix T propagation

        self.children = []  # List of REFERENCES to child vertex objects
                            # CRITICAL: These are references (pointers), NOT deep copies!
                            #
                            # Why references are essential:
                            # -----------------------------
                            # 1. Tree Structure: Forms true parent-child graph in memory
                            # 2. Constraint Propagation: Changes to root's T affect tree traversal
                            # 3. Memory Efficiency: Avoids duplicating entire subtrees
                            # 4. Bidirectional Links: Children can access parent via self.parent
                            #
                            # Usage:
                            # ------
                            # - Empty list [] at initialization (no children yet)
                            # - Populated via add_child() method with vertex references
                            # - Each element points to a vertex object in memory
                            #
                            # WARNING: Do NOT deep copy children when copying a vertex!
                            #          This would break the tree structure.

        self.parent = parent  # Reference to parent vertex (None for root)
                              # NOT deep copied, because this is reference (reference in C++ sense, pointer in C sense)
                              # Forms bidirectional directed tree: parent ↔ children

    def add_child(self, child_vertex):
        """
        Add a child vertex to this vertex and set bidirectional parent-child relationship.

        CRITICAL: Reference-Based Tree Construction
        ===========================================
        This method establishes bidirectional links using REFERENCES (pointers):
        Before call:
        -----------
        self (parent vertex at address 0x1000):
            self.children = [0x2000, 0x3000]  # existing children
        child_vertex (at address 0x4000):
            child_vertex.parent = None  # or some other parent #or this child is a root, we are adding a subtree



        Args:
            child_vertex:  vertex object to add as a child
                           The child represents a hopping derived from this vertex's hopping
                           either through symmetry operation (type="linear")
                           or Hermiticity (type="hermitian")

                           IMPORTANT: child_vertex is NOT deep copied
                                      The REFERENCE to child_vertex is stored in self.children


        Returns:
            None (modifies self.children and child_vertex.parent in-place)

        """
        self.children.append(child_vertex)  # Add REFERENCE to child_vertex to this vertex's children list
                                            # NOT a deep copy - the actual vertex object reference
                                            # After this: self.children[-1] is child_vertex (same object)
                                            #
                                            # Memory effect:
                                            # - self.children list grows by 1 element
                                            # - That element is a reference (memory address) to child_vertex
                                            # - No new vertex object is created

        child_vertex.parent = self  # Set bidirectional relationship: this vertex becomes the child's parent
                                    # Stores new vertex parent's REFERENCE (C++ sense) / POINTER (C sense) to the new vertex
                                    # NOT a deep copy - the actual parent vertex object reference
                                    # After this: child_vertex.parent is self (same object)
                                    #
                                    # Memory effect:
                                    # - child_vertex.parent now points to self's memory address
                                    # - Creates upward link in tree: child → parent
                                    # - Combined with append above: creates bidirectional edge
                                    # WARNING: This overwrites any previous parent!

    def __repr__(self):
        """
        String representation for debugging and display.
        Shows the vertex's role in the tree (ROOT or CHILD), constraint type,
        operation index, parent information, and number of children.
        Returns: str: Compact representation showing vertex type, operation, parent, and children count
                      Format: "vertex(type=<type>, <ROOT/CHILD>, op=<op_idx>, parent=<parent_info>, children=<count>)"

        """
        # Determine if this is a root or child vertex
        root_str = "ROOT" if self.is_root else "CHILD"

        # Show parent's operation index if parent exists, otherwise "None"
        # Parent is None for root vertices
        parent_str = "None" if self.parent is None else f"op={self.parent.hopping.operation_idx}"
        # Return formatted string with key vertex information:
        # - type: constraint type (None, "linear", or "hermitian")
        # - ROOT/CHILD: vertex role in tree
        # - op: this vertex's space group operation index
        # - parent: parent's operation index or "None"
        # - children: number of child vertices
        return (f"vertex(type={self.type}, {root_str}, "
                f"is_root={self.is_root}, "
                f"op={self.hopping.operation_idx}, "
                f"parent={parent_str}, "
                f"children={len(self.children)})")