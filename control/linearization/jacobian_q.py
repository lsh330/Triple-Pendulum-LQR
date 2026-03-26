import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics

# Optimal step for central differences: sqrt(machine_eps) ≈ 1.49e-8
_SQRT_EPS = np.sqrt(np.finfo(np.float64).eps)


def _adaptive_eps(val):
    """Optimal finite-difference step: sqrt(eps_mach) * max(1, |val|)."""
    return _SQRT_EPS * max(1.0, abs(val))


def compute_A_q(q_eq, dq_eq, u_eq, p, eps=None):
    """Central difference of forward_dynamics w.r.t. q -> 4x4 ndarray.

    Uses adaptive per-component step size for near-optimal accuracy when
    eps is None (default).  Pass a scalar eps to override with a fixed step.
    """
    n = len(q_eq)
    A_q = np.zeros((n, n))
    for i in range(n):
        h = _adaptive_eps(q_eq[i]) if eps is None else eps
        q_plus = q_eq.copy()
        q_minus = q_eq.copy()
        q_plus[i] += h
        q_minus[i] -= h
        ddq_plus = forward_dynamics(q_plus, dq_eq, u_eq, p)
        ddq_minus = forward_dynamics(q_minus, dq_eq, u_eq, p)
        A_q[:, i] = (ddq_plus - ddq_minus) / (2.0 * h)
    return A_q
