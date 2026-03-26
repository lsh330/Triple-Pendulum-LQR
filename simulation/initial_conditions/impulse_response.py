import numpy as np
from dynamics.mass_matrix.assembly import mass_matrix


def apply_impulse(q_eq, p, impulse):
    """Compute initial velocities from an impulse applied to the cart.

    Solves M(q_eq) * dq_0 = [impulse, 0, 0, 0].

    Parameters
    ----------
    q_eq : ndarray (4,)
    p : ndarray
        Packed system parameters.
    impulse : float
        Impulse magnitude applied to the cart.

    Returns
    -------
    dq_0 : ndarray (4,)
    """
    M = mass_matrix(q_eq, p)
    rhs = np.zeros(4)
    rhs[0] = impulse
    dq_0 = np.linalg.solve(M, rhs)
    return dq_0
