import numpy as np
from numba import njit


@njit(cache=True)
def pendulum_block(c2, c23, c3, p):
    a1 = p[4]
    a2 = p[5]
    a3 = p[6]
    b1 = p[7]
    b2 = p[8]
    b3 = p[9]

    M11 = a1 + a2 + a3 + 2 * b1 * c2 + 2 * b2 * c23 + 2 * b3 * c3
    M12 = a2 + a3 + b1 * c2 + b2 * c23 + 2 * b3 * c3
    M13 = a3 + b2 * c23 + b3 * c3
    M22 = a2 + a3 + 2 * b3 * c3
    M23 = a3 + b3 * c3
    M33 = a3
    return M11, M12, M13, M22, M23, M33
