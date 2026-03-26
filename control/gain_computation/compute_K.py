import numpy as np


def compute_K(R, B, P):
    """Compute LQR gain matrix K (1x8).

    K = R^{-1} B^T P
    """
    K = np.linalg.solve(R, B.T @ P)
    return K
