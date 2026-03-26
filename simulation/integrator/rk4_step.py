import numpy as np
from numba import njit
from simulation.integrator.state_derivative import state_derivative


@njit(cache=True)
def rk4_step(q, dq, u, p, dt):
    """Perform one RK4 integration step.

    Parameters
    ----------
    q : ndarray (4,)
    dq : ndarray (4,)
    u : float
        Scalar control input.
    p : ndarray
        Packed parameters.
    dt : float

    Returns (q_new, dq_new).
    """
    state = np.empty(8)
    state[:4] = q
    state[4:] = dq

    k1 = state_derivative(state, u, p)
    k2 = state_derivative(state + 0.5 * dt * k1, u, p)
    k3 = state_derivative(state + 0.5 * dt * k2, u, p)
    k4 = state_derivative(state + dt * k3, u, p)

    state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    q_new = state_new[:4].copy()
    dq_new = state_new[4:].copy()
    return q_new, dq_new
