import numpy as np
from numba import njit


@njit(cache=True)
def cart_link_coupling(c1, c12, c123, p):
    mx1 = p[1] * c1 + p[2] * c12 + p[3] * c123
    mx2 = p[2] * c12 + p[3] * c123
    mx3 = p[3] * c123
    return mx1, mx2, mx3
