import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics

_SQRT_EPS = np.sqrt(np.finfo(np.float64).eps)


def _adaptive_eps(val):
    """Optimal finite-difference step: sqrt(eps_mach) * max(1, |val|)."""
    return _SQRT_EPS * max(1.0, abs(val))


def compute_A_dq(q_eq, dq_eq, u_eq, p, eps=None):
    """Central difference of forward_dynamics w.r.t. dq -> 4x4 ndarray.

    Uses adaptive per-component step size for near-optimal accuracy when
    eps is None (default).  Pass a scalar eps to override with a fixed step.
    """
    n = len(dq_eq)
    A_dq = np.zeros((n, n))
    for i in range(n):
        h = _adaptive_eps(dq_eq[i]) if eps is None else eps
        dq_plus = dq_eq.copy()
        dq_minus = dq_eq.copy()
        dq_plus[i] += h
        dq_minus[i] -= h
        ddq_plus = forward_dynamics(q_eq, dq_plus, u_eq, p)
        ddq_minus = forward_dynamics(q_eq, dq_minus, u_eq, p)
        A_dq[:, i] = (ddq_plus - ddq_minus) / (2.0 * h)
    return A_dq
