import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics

_SQRT_EPS = np.sqrt(np.finfo(np.float64).eps)


def _adaptive_eps(val):
    """Optimal finite-difference step: sqrt(eps_mach) * max(1, |val|)."""
    return _SQRT_EPS * max(1.0, abs(val))


def compute_B_u(q_eq, dq_eq, u_eq, p, eps=None):
    """Central difference of forward_dynamics w.r.t. u -> 4x1 ndarray.

    Uses adaptive step size for near-optimal accuracy when eps is None.
    """
    n = len(q_eq)
    h = _adaptive_eps(u_eq) if eps is None else eps
    B_u = np.zeros((n, 1))
    ddq_plus = forward_dynamics(q_eq, dq_eq, u_eq + h, p)
    ddq_minus = forward_dynamics(q_eq, dq_eq, u_eq - h, p)
    B_u[:, 0] = (ddq_plus - ddq_minus) / (2.0 * h)
    return B_u
