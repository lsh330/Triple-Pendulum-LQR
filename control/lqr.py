"""End-to-end LQR gain computation with input validation."""

import numpy as np
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati
from control.gain_computation.compute_K import compute_K


def _validate_cost_matrices(Q, R, n_x, n_u):
    """Validate Q is positive semi-definite and R is positive definite."""
    if Q.shape != (n_x, n_x):
        raise ValueError(f"Q must be ({n_x},{n_x}), got {Q.shape}")
    if R.shape != (n_u, n_u):
        raise ValueError(f"R must be ({n_u},{n_u}), got {R.shape}")

    # Q must be symmetric positive semi-definite
    if not np.allclose(Q, Q.T, atol=1e-12):
        raise ValueError("Q must be symmetric")
    eig_Q = np.linalg.eigvalsh(Q)
    if np.min(eig_Q) < -1e-10:
        raise ValueError(f"Q must be positive semi-definite (min eigenvalue={np.min(eig_Q):.2e})")

    # R must be symmetric positive definite
    if not np.allclose(R, R.T, atol=1e-12):
        raise ValueError("R must be symmetric")
    eig_R = np.linalg.eigvalsh(R)
    if np.min(eig_R) <= 0:
        raise ValueError(f"R must be positive definite (min eigenvalue={np.min(eig_R):.2e})")


def compute_lqr_gains(cfg, Q=None, R=None):
    """Compute LQR gains for the triple inverted pendulum.

    Validates cost matrices and controllability before solving CARE.
    Returns K, A, B, P, Q, R.
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium

    A, B = linearize(q_eq, p)

    if Q is None:
        Q = default_Q()
    if R is None:
        R = default_R()

    n_x = A.shape[0]
    n_u = B.shape[1] if B.ndim == 2 else 1
    _validate_cost_matrices(Q, R, n_x, n_u)

    P = solve_riccati(A, B, Q, R)
    K = compute_K(R, B, P)

    return K, A, B, P, Q, R
