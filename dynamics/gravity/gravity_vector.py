import numpy as np
from numba import njit

from dynamics.trigonometry import compute_trig


@njit(cache=True)
def gravity_vector(q, p):
    trig = compute_trig(q)
    s1 = trig[6]
    s12 = trig[7]
    s123 = trig[8]

    G = np.zeros((4, 1))
    G[0, 0] = 0.0
    G[1, 0] = p[10] * s1 + p[11] * s12 + p[12] * s123
    G[2, 0] = p[11] * s12 + p[12] * s123
    G[3, 0] = p[12] * s123
    return G
