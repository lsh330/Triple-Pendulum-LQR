"""Continuous Algebraic Riccati Equation solver with validation."""

import numpy as np
from scipy.linalg import solve_continuous_are

from utils.logger import get_logger

log = get_logger()


def _check_controllability(A, B, tol=1e-10):
    """Check controllability rank condition: rank([B, AB, ..., A^{n-1}B]) = n."""
    n = A.shape[0]
    C_mat = B.reshape(-1, 1) if B.ndim == 1 else B
    AB = C_mat.copy()
    for _ in range(1, n):
        AB = A @ AB
        C_mat = np.hstack([C_mat, AB])
    rank = np.linalg.matrix_rank(C_mat, tol=tol)
    return rank == n


def solve_riccati(A, B, Q, R):
    """Solve the continuous algebraic Riccati equation with validation.

    Checks controllability of (A,B) and validates the solution P is
    positive definite. Raises ValueError if conditions are not met.

    Returns the solution matrix P.
    """
    B_2d = B.reshape(-1, 1) if B.ndim == 1 else B

    if not _check_controllability(A, B_2d):
        raise ValueError("System (A,B) is not controllable — CARE may not have stabilizing solution")

    try:
        P = solve_continuous_are(A, B_2d, Q, R)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"CARE solver failed: {e}") from e

    # Validate P is positive definite
    eig_min = np.min(np.linalg.eigvalsh(P))
    if eig_min < -1e-10:
        raise ValueError(f"CARE solution P is not positive definite (min eigenvalue={eig_min:.2e})")

    # Enforce exact symmetry
    P = 0.5 * (P + P.T)
    return P
