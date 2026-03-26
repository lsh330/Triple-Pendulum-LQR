import numpy as np


def compute_closed_loop(A, B, K):
    """Compute closed-loop system properties.

    Returns dict with keys: A_cl, poles, is_stable.
    """
    A_cl = A - B @ K
    poles = np.linalg.eigvals(A_cl)
    is_stable = bool(np.all(np.real(poles) < 0))
    return {"A_cl": A_cl, "poles": poles, "is_stable": is_stable}
