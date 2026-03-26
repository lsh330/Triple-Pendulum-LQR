import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


def compute_B_u(q_eq, dq_eq, u_eq, p, eps=1e-7):
    """Central difference of forward_dynamics w.r.t. u -> 4x1 ndarray."""
    n = len(q_eq)
    B_u = np.zeros((n, 1))
    ddq_plus = forward_dynamics(q_eq, dq_eq, u_eq + eps, p)
    ddq_minus = forward_dynamics(q_eq, dq_eq, u_eq - eps, p)
    B_u[:, 0] = (ddq_plus - ddq_minus) / (2.0 * eps)
    return B_u
