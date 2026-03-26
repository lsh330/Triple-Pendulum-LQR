import numpy as np
from numba import njit


@njit(cache=True)
def compute_control(q, dq, q_eq, K_flat):
    """Compute the scalar control input using LQR gain.

    z[:4] = q - q_eq, z[4:] = dq
    u = -K_flat @ z

    Parameters
    ----------
    q : ndarray (4,)
    dq : ndarray (4,)
    q_eq : ndarray (4,)
    K_flat : ndarray (8,)

    Returns
    -------
    u_scalar : float
    """
    z = np.empty(8)
    z[:4] = q - q_eq
    z[4:] = dq
    u_scalar = 0.0
    for i in range(8):
        u_scalar -= K_flat[i] * z[i]
    return u_scalar
