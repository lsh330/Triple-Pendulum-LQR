import numpy as np
from numba import njit


@njit(cache=True)
def assemble_tau(u):
    return np.array([u, 0.0, 0.0, 0.0])
