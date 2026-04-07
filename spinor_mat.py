import numpy as np

# Define Pauli matrices globally for better performance
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I_2x2 = np.eye(2, dtype=complex)


def O3_to_spinor(A, delta=1, tol=1e-7):
    """
    Computes the spinor transformation matrix for a given O(3) matrix A.

    Args:
        A (numpy.ndarray): 3x3 orthogonal matrix in O(3).
        delta (int): 1 for unitary (no time reversal), -1 for antiunitary (time reversal).
        tol (float): Tolerance for floating point comparisons.

    Returns:
        (numpy.ndarray): The 2x2 complex matrix acting on the spinor.
                         If delta == -1, this matrix should be multiplied
                         by the complex conjugate of the spinor.
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

    # Step 5: Apply time reversal if delta == -1
    if delta == 1:
        return U
    elif delta == -1:
        U_conj = np.conj(U)
        T_mat = -1j * sigma_y @ U_conj
        return T_mat
    else:
        raise ValueError("delta must be 1 or -1")