import numpy as np
from control.linearization.jit_jacobians import compute_jacobians_jit


def linearize(q_eq, p, eps=None):
    """Linearize the system about the equilibrium point.

    Returns (A, B) where A is 8x8 and B is 8x1.

    Parameters
    ----------
    q_eq : ndarray (4,)
        Configuration at the linearization point.
    p : ndarray
        Packed system parameters.
    eps : float or None
        Kept for API compatibility but ignored; the JIT version always
        uses adaptive per-component steps.
    """
    dq_eq = np.zeros(4)
    u_eq = 0.0

    A, B = compute_jacobians_jit(q_eq, dq_eq, u_eq, p)
    return A, B
