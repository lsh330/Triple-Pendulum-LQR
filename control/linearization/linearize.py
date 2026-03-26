"""Linearize the triple-pendulum system about an operating point.

Uses analytical Jacobians (exact dG/dq, dM/dq, M^{-1}) for A_q and B_u,
with numerical central differences only for A_dq (Coriolis velocity term).
Falls back to fully numerical JIT Jacobians if analytical computation fails.
"""

import numpy as np
from control.linearization.analytical_jacobian import compute_analytical_jacobians
from control.linearization.jit_jacobians import compute_jacobians_jit


def linearize(q_eq, p, eps=None, method="analytical"):
    """Linearize the system about the equilibrium point.

    Returns (A, B) where A is 8x8 and B is 8x1.

    Parameters
    ----------
    q_eq : ndarray (4,)
        Configuration at the linearization point.
    p : ndarray
        Packed system parameters.
    eps : float or None
        Kept for API compatibility; ignored by both methods.
    method : str
        "analytical" (default, exact dG/dq + dM/dq) or "numerical" (JIT FD).
    """
    if method == "analytical":
        try:
            A, B = compute_analytical_jacobians(q_eq, p)
            return A, B
        except Exception:
            pass

    # Fallback: JIT numerical Jacobians
    dq_eq = np.zeros(4)
    u_eq = 0.0
    A, B = compute_jacobians_jit(q_eq, dq_eq, u_eq, p)
    return A, B
