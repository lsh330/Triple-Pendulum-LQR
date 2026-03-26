import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


def compute_A_dq(q_eq, dq_eq, u_eq, p, eps=1e-7):
    """Central difference of forward_dynamics w.r.t. dq -> 4x4 ndarray."""
    n = len(dq_eq)
    A_dq = np.zeros((n, n))
    for i in range(n):
        dq_plus = dq_eq.copy()
        dq_minus = dq_eq.copy()
        dq_plus[i] += eps
        dq_minus[i] -= eps
        ddq_plus = forward_dynamics(q_eq, dq_plus, u_eq, p)
        ddq_minus = forward_dynamics(q_eq, dq_minus, u_eq, p)
        A_dq[:, i] = (ddq_plus - ddq_minus) / (2.0 * eps)
    return A_dq
