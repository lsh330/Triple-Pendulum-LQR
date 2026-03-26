import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


def compute_A_q(q_eq, dq_eq, u_eq, p, eps=1e-7):
    """Central difference of forward_dynamics w.r.t. q -> 4x4 ndarray."""
    n = len(q_eq)
    A_q = np.zeros((n, n))
    for i in range(n):
        q_plus = q_eq.copy()
        q_minus = q_eq.copy()
        q_plus[i] += eps
        q_minus[i] -= eps
        ddq_plus = forward_dynamics(q_plus, dq_eq, u_eq, p)
        ddq_minus = forward_dynamics(q_minus, dq_eq, u_eq, p)
        A_q[:, i] = (ddq_plus - ddq_minus) / (2.0 * eps)
    return A_q
