import numpy as np
from numba import njit

from dynamics.trigonometry import compute_trig
from dynamics.mass_matrix.cart_link_coupling import cart_link_coupling
from dynamics.mass_matrix.pendulum_block import pendulum_block


@njit(cache=True)
def mass_matrix(q, p):
    c1, c12, c123, c2, c23, c3, s1, s12, s123 = compute_trig(q)
    mx1, mx2, mx3 = cart_link_coupling(c1, c12, c123, p)
    M11, M12, M13, M22, M23, M33 = pendulum_block(c2, c23, c3, p)

    M = np.zeros((4, 4))
    # Row 0: cart
    M[0, 0] = p[0]
    M[0, 1] = mx1
    M[0, 2] = mx2
    M[0, 3] = mx3
    # Symmetric
    M[1, 0] = mx1
    M[2, 0] = mx2
    M[3, 0] = mx3
    # Pendulum block
    M[1, 1] = M11
    M[1, 2] = M12
    M[1, 3] = M13
    M[2, 1] = M12
    M[2, 2] = M22
    M[2, 3] = M23
    M[3, 1] = M13
    M[3, 2] = M23
    M[3, 3] = M33
    return M
