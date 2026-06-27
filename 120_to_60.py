import re
import numpy as np

# The Master Regex from your script
term_pattern = re.compile(r"([+-]?)\s*([xyz])|([+-]?)\s*(\d+(?:[./]\d+)?)", re.IGNORECASE)

def parse_single_expression(expression_str):
    """
    Parses a string like '-x+1/2' or 'x-y' into numerical coefficients.
    Returns: (coeff_x, coeff_y, coeff_z, translation)
    """
    c_x, c_y, c_z, trans = 0.0, 0.0, 0.0, 0.0

    for match in term_pattern.finditer(expression_str):
        # --- CASE 1: It's a Variable: x, y, z ---
        if match.group(2):
            sign_str = match.group(1)
            variable = match.group(2).lower()
            value = -1.0 if sign_str == '-' else 1.0

            if variable == 'x': c_x += value
            elif variable == 'y': c_y += value
            elif variable == 'z': c_z += value

        # --- CASE 2: It's a Number: Translation ---
        elif match.group(4):
            sign_str = match.group(3)
            number_str = match.group(4)
            if '/' in number_str:
                num, den = number_str.split('/')
                val = float(num) / float(den)
            else:
                val = float(number_str)
            if sign_str == '-': val = -val
            trans += val

    return c_x, c_y, c_z, trans

def format_expression(row):
    """Converts a row of coefficients [c_x, c_y] back into a readable string."""
    terms = []
    if row[0] == 1: terms.append('x')
    elif row[0] == -1: terms.append('-x')
    elif row[0] not in (0, 1, -1): terms.append(f"{int(row[0])}x")

    if row[1] == 1: terms.append('+y' if terms else 'y')
    elif row[1] == -1: terms.append('-y')
    elif row[1] not in (0, 1, -1): terms.append(f"{int(row[1]):+}y")

    if not terms: return '0'
    return ''.join(terms).lstrip('+')

# The transformation matrix A and its inverse A_inv
A = np.array([
    [1, 1],
    [0, 1]
])

A_inv = np.array([
    [1, -1],
    [0,  1]
])

# The symmetry block from the FINDSYM CIF file
cif_symmetry_block = """
1 x,y,z
2 -y,x-y,z
3 -x+y,-x,z
4 x,x-y,-z
5 -x+y,y,-z
6 -y,-x,-z
7 -x+y,-x,-z
8 x,y,-z
9 -y,x-y,-z
10 -x+y,y,z
11 -y,-x,z
12 x,x-y,z
"""

output_filename = "transformations_60.txt"

# Open the file for writing
with open(output_filename, "w") as f:

    # Write the table header
    header = f"{'ID':<3} | {'Old Symmetry (120°)':<15} | {'S Matrix (Old)':<15} | {'New Symmetry (60°)'}"
    f.write(header + "\n")
    f.write("-" * 70 + "\n")
    print(header)
    print("-" * 70)

    # List to store just the new operations for a clean CIF-like output later
    new_operations_clean = []

    for line in cif_symmetry_block.strip().split('\n'):
        line = line.strip()
        if not line: continue

        # Split by whitespace to separate the ID from the xyz string
        parts = line.split()
        sym_id = parts[0]
        xyz_str = parts[1]

        # Split the symmetry operation into x, y, and z parts
        ops = xyz_str.split(',')
        op_x, op_y, op_z = ops[0], ops[1], ops[2]

        # 1. Use the regex parser to extract coefficients
        cx1, cy1, cz1, tr1 = parse_single_expression(op_x)
        cx2, cy2, cz2, tr2 = parse_single_expression(op_y)

        # Build the 2x2 matrix S for the x,y plane
        S = np.array([
            [cx1, cy1],
            [cx2, cy2]
        ])

        # 2. Compute the new transformation matrix using A^{-1} * S * A
        S_new = np.rint(A_inv @ S @ A).astype(int)

        # 3. Format the new matrix back into a string
        new_op_x = format_expression(S_new[0])
        new_op_y = format_expression(S_new[1])
        new_xyz_str = f"{new_op_x},{new_op_y},{op_z}"

        # Format the S matrix for printing
        s_matrix_str = f"[{int(S[0][0]):>2} {int(S[0][1]):>2}; {int(S[1][0]):>2} {int(S[1][1]):>2}]"

        # Create the row string
        row_str = f"{sym_id:<3} | {xyz_str:<15} | {s_matrix_str:<15} | {new_xyz_str}"

        # Write to file and print to console
        f.write(row_str + "\n")
        print(row_str)

        # Save the clean version
        new_operations_clean.append(f"{sym_id} {new_xyz_str}")

    # Write a clean, CIF-ready block at the end of the file
    f.write("\n\n--- Clean CIF Format (60° Setting) ---\n")
    for op in new_operations_clean:
        f.write(op + "\n")

print(f"\nSuccess! The transformations have been written to '{output_filename}'.")