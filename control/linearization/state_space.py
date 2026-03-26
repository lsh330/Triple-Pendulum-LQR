import numpy as np


def assemble_state_space(A_q, A_dq, B_u):
    """Assemble full state-space matrices A (8x8) and B (8x1).

    A = [[0, I], [A_q, A_dq]]
    B = [[0], [B_u]]
    """
    n = A_q.shape[0]
    Z = np.zeros((n, n))
    I = np.eye(n)

    A = np.block([
        [Z, I],
        [A_q, A_dq]
    ])

    B = np.vstack([
        np.zeros((n, 1)),
        B_u
    ])

    return A, B
