import numpy as np
from numba import njit
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def state_derivative(state_8, u, p):
    """Compute the time derivative of the full state vector.

    state_8[:4] = q, state_8[4:] = dq
    dstate[:4] = dq, dstate[4:] = forward_dynamics(q, dq, u, p)

    Parameters
    ----------
    state_8 : ndarray (8,)
    u : float
        Scalar control input.
    p : ndarray
        Packed parameters.
    """
    q = state_8[:4]
    dq = state_8[4:]
    dstate = np.empty(8)
    dstate[:4] = dq
    dstate[4:] = forward_dynamics(q, dq, u, p)
    return dstate
