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



    def get_numpy_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix as a numpy Matrix

        Args:
            operation_idx: index of the space group operation

        Returns:
            np array: representation matrix for this atom's orbitals
        """
        return self.get_representation_matrix(operation_idx)
        # return sp.Matrix(self.get_representation_matrix(operation_idx))


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
        self.T_reconstructed = None  # reconstructed by all constraints

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



class T_tilde_total():
    """
     this class stores the k-space Hamiltonian blocks, and constructs the total k-space Hamiltonian
    """

    def __init__(self, unit_cell_atoms):
        """

        Args:
            unit_cell_atoms:  unit_cell_atoms contain constructed T blocks, it is deep-copied to decouple
        """
        self.unit_cell_atoms = deepcopy(unit_cell_atoms)
        # Flattened dictionary to store all T_tilde_val blocks from all atoms
        # Key: (to_atom_id, from_atom_id), Value: SymPy Matrix
        self.T_tilde_from_unit_cell_atoms = {}
        # Iterate through each atom in the unit cell
        for atom in self.unit_cell_atoms:
            # Iterate through the T_tilde_val dictionary of the atom
            # atom.T_tilde_val contains the summed k-space hopping matrices
            for key, matrix in atom.T_tilde_val.items():
                # Add the matrix to the flattened dictionary
                self.T_tilde_from_unit_cell_atoms[key] = matrix

        self.complete_hermitian_blocks()
        # Initialize attributes for block matrix construction
        self.total_hamiltonian = None  # The final assembled matrix
        self.hamiltonian_dimension = None  # Total dimension of the Hamiltonian
        self.sorted_wyckoff_instance_ids = None
        self.block_dimensions = None
        self.system_name = self.unit_cell_atoms[0].parsed_config["name"]



    def complete_hermitian_blocks(self):
        """
        Iterates through all atoms and ensures that for every block H_ij,
        the corresponding H_ji = H_ij^dagger exists in the total Hamiltonian.
        Returns:

        """
        for atom in self.unit_cell_atoms:
            # Iterate through the T_tilde_val dictionary of the atom
            # key is (this_wyckoff_instance_id, other_wyckoff_instance_id)
            for (this_id, other_id), matrix in atom.T_tilde_val.items():
                # Swap the key to represent the conjugate block
                swapped_key = (other_id, this_id)
                # Check if the swapped key exists in the master dictionary
                if swapped_key not in self.T_tilde_from_unit_cell_atoms:
                    # If not, add it with the Hermitian conjugate of the matrix
                    self.T_tilde_from_unit_cell_atoms[swapped_key] = matrix.H


    def sort_wyckoff_instance_ids(self):
        """
        Sort wyckoff_instance_ids using natural sorting that respects the structure:
        <Element><WyckoffPosition>_<AtomIndex>
        Format examples:
        - 'C0_0': Carbon, Wyckoff position 0, atom index 0
        - 'Ca0_1': Calcium, Wyckoff position 0, atom index 1
        - 'Ca1_0': Calcium, Wyckoff position 1, atom index 0

        Sorting priority:
        1. Element symbol (alphabetically)
        2. Wyckoff position (numerically)
        3. Atom index (numerically)

        Examples:
            Input:  ['Ca0_1', 'C0_0', 'Ca1_0', 'Ca0_0', 'C0_1']
            Output: ['C0_0', 'C0_1', 'Ca0_0', 'Ca0_1', 'Ca1_0']
            Grouped by element → position → index:
            C:  position 0: [C0_0, C0_1]
            Ca: position 0: [Ca0_0, Ca0_1]
                position 1: [Ca1_0]

        Returns:
            list: Naturally sorted list of wyckoff_instance_ids
        TODO: name convention in conf file should be optimized

        """

        def parse_wyckoff_id(wyckoff_id):
            """
            Parse wyckoff_instance_id into (element, position, index).
            Args:
                wyckoff_id: String like 'Ca0_1', 'C0_0', 'N1_0'

            Returns:
                 tuple: (element_str, position_int, index_int)
            Examples:
                'Ca0_1' → ('Ca', 0, 1)
                'C0_0'  → ('C', 0, 0)
                'N1_0'  → ('N', 1, 0)
            """
            # Match: one or more letters, followed by digits, underscore, digits
            # Example: Ca0_1 -> Group 1: Ca, Group 2: 0, Group 3: 1
            match = re.match(r'^([A-Za-z]+)(\d+)_(\d+)$', wyckoff_id)

            if not match:
                raise ValueError(f"Invalid wyckoff_instance_id format: {wyckoff_id}. Expected format like 'Ca0_1'")

            element = match.group(1)
            position = int(match.group(2))  # Wyckoff position index
            index = int(match.group(3))  # Atom index within that Wyckoff position

            return (element, position, index)

        # Extract unique wyckoff_instance_ids from unit_cell_atoms
        wyckoff_instance_ids = [atom.wyckoff_instance_id for atom in self.unit_cell_atoms]
        # Sort by (element, position, index) tuple
        sorted_ids = sorted(wyckoff_instance_ids, key=parse_wyckoff_id)
        self.sorted_wyckoff_instance_ids = sorted_ids
        return sorted_ids

    def construct_total_hamiltonian(self):
        """
        Construct the total k-space Hamiltonian matrix from block matrices.
        The Hamiltonian is block-structured based on wyckoff_instance_ids:

        H_total = | H_00  H_01  H_02  ... |
                 | H_10  H_11  H_12  ... |
                | H_20  H_21  H_22  ... |
               | ...   ...   ...   ... |
        where H_ij is the hopping block from atom j to atom i.

        This method directly assembles the block matrices using SymPy's block matrix
        construction

        Workflow:
        1. Sort wyckoff_instance_ids to establish consistent ordering
        2. Build a 2D list of block matrices
        3. Use sp.BlockMatrix to construct the total Hamiltonian

        Returns:
             sympy.Matrix: Complete k-space Hamiltonian matrix
        """
        # Step 1: Sort wyckoff_instance_ids to establish row/column ordering
        self.sort_wyckoff_instance_ids()
        # Step 2: Calculate block dimensions for each wyckoff_instance_id
        block_dimensions = {}
        for atom in self.unit_cell_atoms:
            wyckoff_id = atom.wyckoff_instance_id
            num_orbitals = atom.num_orbitals
            block_dimensions[wyckoff_id] = num_orbitals
        # print(f"block_dimensions={block_dimensions}")
        # Step 3: Calculate tot al Hamiltonian dimension
        self.block_dimensions = block_dimensions
        self.hamiltonian_dimension = sum(block_dimensions.values())
        # print(f"self.hamiltonian_dimension={self.hamiltonian_dimension}")
        # Step 4: Build 2D list of block matrices
        # blocks[i][j] corresponds to H[to_id_i, from_id_j]
        # n_atoms = len(self.sorted_wyckoff_instance_ids)
        blocks = []
        for i, to_id in enumerate(self.sorted_wyckoff_instance_ids):
            row_blocks = []
            for j, from_id in enumerate(self.sorted_wyckoff_instance_ids):
                key = (to_id, from_id)
                # Check if block exists in dictionary
                if key in self.T_tilde_from_unit_cell_atoms:
                    # Use existing block matrix
                    block = self.T_tilde_from_unit_cell_atoms[key]
                else:
                    # Required block is missing - this means the block is 0
                    n_rows = self.block_dimensions[to_id]
                    n_cols = self.block_dimensions[from_id]
                    # Create a zero matrix of the correct size
                    block = sp.zeros(n_rows, n_cols)

                row_blocks.append(block)
            blocks.append(row_blocks)

        block_matrix = sp.BlockMatrix(blocks)
        # Step 6: Convert BlockMatrix to regular Matrix for easier manipulation
        self.total_hamiltonian = sp.Matrix(block_matrix)
        self.total_hamiltonian = sp.expand(sp.simplify(self.total_hamiltonian))
        return self.total_hamiltonian


    def _fix_latex_subscripts(self, latex_str):
        r"""
        Fix LaTeX double-subscript errors by converting re_XXX and im_XXX to Re(XXX) and Im(XXX).

        Transforms:
            re_T^{0}_{2s,2s} → \operatorname{Re}(T^{0}_{2s,2s})
            im_T^{0}_{2s,2s} → \operatorname{Im}(T^{0}_{2s,2s})
            \overline{re_T^{0}_{2s,2s}} → \overline{\operatorname{Re}(T^{0}_{2s,2s})}
            \operatorname{re} → \operatorname{Re}
            \operatorname{im} → \operatorname{Im}

        Args:
            latex_str: LaTeX string with potential re_XXX and im_XXX symbols

        Returns:
            Fixed LaTeX string
        """
        import re

        # Pattern to match re_SYMBOL or im_SYMBOL where SYMBOL can contain ^{...} and _{...}
        # This captures the entire symbol including superscripts and subscripts

        # Match re_ or im_ followed by anything up to the next space, operator, or delimiter
        # We need to be careful to capture the full symbol including ^{...} and _{...}

        def replace_re_im(match):
            """Helper function to replace matched re_/im_ patterns"""
            prefix = match.group(1)  # 're' or 'im'
            symbol = match.group(2)  # The rest of the symbol

            # Determine the LaTeX operator name
            if prefix == 're':
                op_name = r'\operatorname{Re}'
            else:  # prefix == 'im'
                op_name = r'\operatorname{Im}'

            return f"{op_name}({symbol})"

        # Pattern explanation:
        # (re|im)_          : Match 're_' or 'im_'
        # ([A-Za-z]+        : Start capturing: one or more letters (symbol name)
        # (?:\^{[^}]+})*    : Zero or more superscripts ^{...}
        # (?:_{[^}]+})*     : Zero or more subscripts _{...}
        # )                 : End capturing
        pattern = r'(re|im)_([A-Za-z]+(?:\^{[^}]+})*(?:_{[^}]+})*)'

        # Apply replacement
        fixed_str = re.sub(pattern, replace_re_im, latex_str)

        # Fix SymPy's standard output for real/imaginary parts
        # SymPy outputs \operatorname{re} and \operatorname{im} (lowercase)
        # We replace them with \operatorname{Re} and \operatorname{Im} (uppercase)
        fixed_str = fixed_str.replace(r'\operatorname{re}', r'\operatorname{Re}')
        fixed_str = fixed_str.replace(r'\operatorname{im}', r'\operatorname{Im}')

        return fixed_str


    def write_hamiltonian_to_latex(self, filename, precision=3):
        H = sp.simplify(self.total_hamiltonian)
        # Convert to LaTeX using SymPy's latex() function

        H = self.round_matrix_coefficients(H, precision)
        H = sp.expand(H)
        latex_str = sp.latex(H, mat_delim='[')
        latex_str = self._fix_latex_subscripts(latex_str)
        # Write to file
        with open(filename, 'w') as f:
            f.write(latex_str)

    def round_matrix_coefficients(self, matrix, precision):
        """
        Round all numerical coefficients in a symbolic matrix to specified precision.
        """
        rows, cols = matrix.shape
        rounded_matrix = sp.zeros(rows, cols)

        for i in range(rows):
            for j in range(cols):
                rounded_matrix[i, j] = self.round_expression_coefficients(matrix[i, j], precision)

        return rounded_matrix


    def round_expression_coefficients(self, expr, precision):
        """
        Round numerical coefficients in a SymPy expression to specified precision.

        This recursively processes the entire expression tree.
        """
        if expr == 0 or expr is sp.S.Zero:
            return sp.S.Zero

        # If it's a number, round it directly
        if expr.is_Number:
            return self._round_number(expr, precision)

        # If it's an addition, round each term separately
        if isinstance(expr, sp.Add):
            rounded_terms = []
            for term in expr.args:
                rounded_term = self.round_expression_coefficients(term, precision)
                if rounded_term != sp.S.Zero:  # Skip zero terms
                    rounded_terms.append(rounded_term)

            if len(rounded_terms) == 0:
                return sp.S.Zero
            elif len(rounded_terms) == 1:
                return rounded_terms[0]
            else:
                return sp.Add(*rounded_terms)

        # If it's a multiplication, process it specially
        if isinstance(expr, sp.Mul):
            return self._round_multiplication(expr, precision)

        # If it's a power, process base and exponent
        if isinstance(expr, sp.Pow):
            base_rounded = self.round_expression_coefficients(expr.base, precision)
            exp_rounded = self.round_expression_coefficients(expr.exp, precision)
            return base_rounded ** exp_rounded

        # For other types (like Symbol), return as-is
        return expr


    def _round_number(self, num, precision):
        """Helper to round a SymPy number to specified precision."""
        threshold = 10 ** (-precision - 2)  # Threshold for treating as zero

        if num.is_real:
            val = float(num)
            if abs(val) < threshold:
                return sp.S.Zero
            rounded_val = round(val, precision)
            # Return as integer if it's a whole number
            if abs(rounded_val - round(rounded_val)) < threshold:
                return sp.Integer(int(round(rounded_val)))
            return sp.Float(rounded_val, precision)

        elif num.is_complex:
            real_part = float(sp.re(num))
            imag_part = float(sp.im(num))

            # Round and check threshold for real part
            if abs(real_part) < threshold:
                real_rounded = 0.0
            else:
                real_rounded = round(real_part, precision)

            # Round and check threshold for imaginary part
            if abs(imag_part) < threshold:
                imag_rounded = 0.0
            else:
                imag_rounded = round(imag_part, precision)

            # Construct the result
            if real_rounded == 0.0 and imag_rounded == 0.0:
                return sp.S.Zero
            elif imag_rounded == 0.0:
                if abs(real_rounded - round(real_rounded)) < threshold:
                    return sp.Integer(int(round(real_rounded)))
                return sp.Float(real_rounded, precision)
            elif real_rounded == 0.0:
                if abs(imag_rounded - round(imag_rounded)) < threshold:
                    return sp.I * sp.Integer(int(round(imag_rounded)))
                return sp.I * sp.Float(imag_rounded, precision)
            else:
                result = sp.S.Zero
                if abs(real_rounded - round(real_rounded)) < threshold:
                    result += sp.Integer(int(round(real_rounded)))
                else:
                    result += sp.Float(real_rounded, precision)

                if abs(imag_rounded - round(imag_rounded)) < threshold:
                    result += sp.I * sp.Integer(int(round(imag_rounded)))
                else:
                    result += sp.I * sp.Float(imag_rounded, precision)
                return result

        else:
            return num

    def _round_multiplication(self, expr, precision):
        """
        Round coefficients in a multiplication expression.

        Strategy: Extract ALL numerical factors, round their product,
        then multiply by all symbolic factors.
        """
        # Separate all arguments into numerical and symbolic
        numerical_factors = []
        symbolic_factors = []

        for arg in expr.args:
            if arg.is_Number:
                numerical_factors.append(arg)
            else:
                # Recursively process non-numerical arguments
                # (they might contain nested multiplications with numbers)
                processed_arg = self.round_expression_coefficients(arg, precision)
                if processed_arg != sp.S.Zero:
                    symbolic_factors.append(processed_arg)

        # Multiply all numerical factors together
        if len(numerical_factors) == 0:
            numerical_coeff = sp.S.One
        else:
            numerical_coeff = sp.S.One
            for num in numerical_factors:
                numerical_coeff *= num

        # Round the combined numerical coefficient
        rounded_coeff = self._round_number(numerical_coeff, precision)

        # If coefficient rounded to zero, return zero
        if rounded_coeff == sp.S.Zero:
            return sp.S.Zero

        # Reconstruct the expression
        if len(symbolic_factors) == 0:
            # Only numerical part
            return rounded_coeff
        elif rounded_coeff == sp.S.One:
            # Only symbolic part
            if len(symbolic_factors) == 1:
                return symbolic_factors[0]
            else:
                return sp.Mul(*symbolic_factors)
        else:
            # Both numerical and symbolic parts
            if len(symbolic_factors) == 1:
                return rounded_coeff * symbolic_factors[0]
            else:
                return rounded_coeff * sp.Mul(*symbolic_factors)

    def create_parameter_input_file(self, filename):
        """
        Create a text file for user to input hopping parameter values.
        Extracts all free parameters from the Hamiltonian and creates an input template.

        Format:
            re_T_xxx=
            im_T_xxx=
        Real and imaginary parts of the same parameter are grouped together,
        with a dashed line separating different base parameters.

        Args:
            filename: Path to the output text file

        Returns:
            dict: Information about the parameters written
            {
                're_params': list,
                'im_params': list
            }
        """
        if self.total_hamiltonian is None:
            raise ValueError("Hamiltonian not constructed yet. Call construct_total_hamiltonian() first.")

        # Extract all free symbols from the Hamiltonian
        all_symbols = self.total_hamiltonian.free_symbols

        # Separate symbols into categories
        hopping_params = []
        k_params = set()  # k0, k1, k2 (exclude these)

        for symbol in all_symbols:
            symbol_str = str(symbol)
            # Skip k-vector symbols
            if symbol_str in ['k0', 'k1', 'k2']:
                k_params.add(symbol_str)
                continue
            # Check if it's a hopping parameter (real or imaginary)
            if symbol_str.startswith('re_T') or symbol_str.startswith('im_T'):
                hopping_params.append(symbol_str)

        # Custom sorting function to group re_ and im_ parts together
        def sort_key(param_name):
            # Extract the base name by removing the prefix
            if param_name.startswith('re_'):
                base_name = param_name[3:]
                prefix_order = 0  # Real parts come first
            elif param_name.startswith('im_'):
                base_name = param_name[3:]
                prefix_order = 1  # Imaginary parts come second
            else:
                base_name = param_name
                prefix_order = 2

            # Sort primarily by the base name, secondarily by the prefix order
            return (base_name, prefix_order)

        # Sort parameters using the custom key
        hopping_params_sorted = sorted(hopping_params, key=sort_key)

        # Separate them back for the summary/return dictionary
        re_params_sorted = [p for p in hopping_params_sorted if p.startswith('re_')]
        im_params_sorted = [p for p in hopping_params_sorted if p.startswith('im_')]

        # Write to file
        with open(filename, 'w') as f:
            f.write("# Hopping Parameter Input File\n")
            f.write("# Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("#\n")
            f.write("# Instructions:\n")
            f.write("# 1. Fill in numerical values after the '=' symbol\n")
            f.write("# 2. Use floating point numbers (e.g., 1.0, -2.5, 0.0)\n")
            f.write("# 3. Lines starting with '#' are comments and will be ignored\n")
            f.write("# 4. Do not modify the parameter names (left side of '=')\n")
            f.write("#\n")
            f.write(f"#   - Independent real parts (re_T): {len(re_params_sorted)}\n")
            f.write(f"#   - Independent imaginary parts (im_T): {len(im_params_sorted)}\n")
            f.write("#\n")
            f.write("# " + "=" * 70 + "\n\n")

            # Write all parameters grouped together
            if hopping_params_sorted:
                f.write("# " + "=" * 70 + "\n")
                f.write("# Hopping Parameters (Real and Imaginary Parts)\n")
                f.write("# " + "=" * 70 + "\n\n")

                previous_base = None
                for param in hopping_params_sorted:
                    # Extract the base name to check for changes
                    if param.startswith('re_'):
                        current_base = param[3:]
                    elif param.startswith('im_'):
                        current_base = param[3:]
                    else:
                        current_base = param

                    # If the base name changed and it's not the first parameter, write a separator
                    if previous_base is not None and current_base != previous_base:
                        f.write("# " + "-" * 70 + "\n")

                    f.write(f"{param}=\n")
                    previous_base = current_base
                f.write("\n")

        # Print summary
        print(f"\n✓ Created parameter input file: {filename}")
        print(f"\nParameter summary:")
        print(f"  - Independent real parts (re_T): {len(re_params_sorted)}")
        print(f"  - Independent imaginary parts (im_T): {len(im_params_sorted)}")

        # Return summary information
        return {
            're_params': re_params_sorted,
            'im_params': im_params_sorted
        }