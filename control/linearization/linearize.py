import numpy as np
from control.linearization.jacobian_q import compute_A_q
from control.linearization.jacobian_dq import compute_A_dq
from control.linearization.jacobian_u import compute_B_u
from control.linearization.state_space import assemble_state_space


def linearize(q_eq, p, eps=1e-5):
    """Linearize the system about the equilibrium point.

    Returns (A, B) where A is 8x8 and B is 8x1.
    """
    dq_eq = np.zeros(4)
    u_eq = 0.0  # scalar input

    A_q = compute_A_q(q_eq, dq_eq, u_eq, p, eps)
    A_dq = compute_A_dq(q_eq, dq_eq, u_eq, p, eps)
    B_u = compute_B_u(q_eq, dq_eq, u_eq, p, eps)

    A, B = assemble_state_space(A_q, A_dq, B_u)
    return A, B
