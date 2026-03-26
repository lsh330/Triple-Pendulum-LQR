import numpy as np
from numba import njit


@njit(cache=True)
def solve_acceleration(M, rhs):
    return np.linalg.solve(M, rhs)
