import numpy as np
from numba import njit

from dynamics.mass_matrix.assembly import mass_matrix


@njit(cache=True)
def coriolis_vector(q, dq, p):
    n = 4
    eps = 1e-5
    h = np.zeros((n, 1))

    # Precompute dM/dq_k via central differences
    dMdq = np.zeros((n, n, n))  # dMdq[i, j, k] = dM_ij / dq_k
    for k in range(n):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[k] += eps
        q_minus[k] -= eps
        M_plus = mass_matrix(q_plus, p)
        M_minus = mass_matrix(q_minus, p)
        for i in range(n):
            for j in range(n):
                dMdq[i, j, k] = (M_plus[i, j] - M_minus[i, j]) / (2.0 * eps)

    for i in range(n):
        val = 0.0
        for j in range(n):
            for k in range(n):
                c_ijk = 0.5 * (dMdq[i, j, k] + dMdq[i, k, j] - dMdq[j, k, i])
                val += c_ijk * dq[j] * dq[k]
        h[i, 0] = val
    return h
