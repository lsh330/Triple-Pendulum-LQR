"""Analytical Coriolis vector C(q,dq)*dq via closed-form partial derivatives of M."""

import numpy as np
from numba import njit

from dynamics.trigonometry import compute_trig


@njit(cache=True)
def coriolis_vector(q, dq, p):
    """Compute C(q,dq)*dq analytically. Returns (4,1) array.

    ∂M/∂q is derived analytically from the mass matrix structure.
    M depends on q through: sin/cos of th1, th1+th2, th1+th2+th3, th2, th2+th3, th3.
    ∂M/∂x = 0 (M does not depend on cart position).
    """
    c1, c12, c123, c2, c23, c3, s1, s12, s123 = compute_trig(q)
    gx1, gx2, gx3 = p[1], p[2], p[3]
    b1, b2, b3 = p[7], p[8], p[9]

    dx, d1, d2, d3 = dq[0], dq[1], dq[2], dq[3]

    # ∂M/∂x = 0 everywhere

    # ∂M/∂th1: only cart-link coupling row/col changes
    # ∂mx1/∂th1 = -gx1*s1 - gx2*s12 - gx3*s123
    # ∂mx2/∂th1 = -gx2*s12 - gx3*s123
    # ∂mx3/∂th1 = -gx3*s123
    # Pendulum block: ∂M[1:,1:]/∂th1 = 0
    dmx1_d1 = -gx1*s1 - gx2*s12 - gx3*s123
    dmx2_d1 = -gx2*s12 - gx3*s123
    dmx3_d1 = -gx3*s123

    # ∂M/∂th2:
    # ∂mx1/∂th2 = -gx2*s12 - gx3*s123
    # ∂mx2/∂th2 = -gx2*s12 - gx3*s123
    # ∂mx3/∂th2 = -gx3*s123
    # Pendulum block ∂/∂th2: only terms with c2, c23
    # ∂M11/∂th2 = -2*b1*s2 - 2*b2*s23
    # ∂M12/∂th2 = -b1*s2 - b2*s23
    # ∂M13/∂th2 = -b2*s23
    # ∂M22/∂th2 = 0, ∂M23/∂th2 = 0, ∂M33/∂th2 = 0
    s2 = np.sin(q[2])
    s23_rel = np.sin(q[2] + q[3])
    s3 = np.sin(q[3])

    dmx1_d2 = -gx2*s12 - gx3*s123
    dmx2_d2 = -gx2*s12 - gx3*s123
    dmx3_d2 = -gx3*s123
    dM11_d2 = -2*b1*s2 - 2*b2*s23_rel
    dM12_d2 = -b1*s2 - b2*s23_rel
    dM13_d2 = -b2*s23_rel

    # ∂M/∂th3:
    # ∂mx1/∂th3 = -gx3*s123
    # ∂mx2/∂th3 = -gx3*s123
    # ∂mx3/∂th3 = -gx3*s123
    # Pendulum block ∂/∂th3: terms with c23, c3
    # ∂M11/∂th3 = -2*b2*s23 - 2*b3*s3
    # ∂M12/∂th3 = -b2*s23 - 2*b3*s3
    # ∂M13/∂th3 = -b2*s23 - b3*s3
    # ∂M22/∂th3 = -2*b3*s3
    # ∂M23/∂th3 = -b3*s3
    # ∂M33/∂th3 = 0
    dmx1_d3 = -gx3*s123
    dmx2_d3 = -gx3*s123
    dmx3_d3 = -gx3*s123
    dM11_d3 = -2*b2*s23_rel - 2*b3*s3
    dM12_d3 = -b2*s23_rel - 2*b3*s3
    dM13_d3 = -b2*s23_rel - b3*s3
    dM22_d3 = -2*b3*s3
    dM23_d3 = -b3*s3

    # Pack ∂M_ij/∂q_k into a 4x4x4 tensor (only nonzero entries)
    # dMdq[i,j,k] = ∂M[i,j]/∂q[k]
    # q[0]=x, q[1]=th1, q[2]=th2, q[3]=th3

    # Christoffel: h_i = sum_{j,k} 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * dq_j * dq_k
    # Since dM/dx = 0, only k=1,2,3 contribute

    h = np.zeros(4)

    # Build dM/dq_k for k=1,2,3 as 4x4 matrices (sparse)
    # k=1: ∂M/∂th1
    dM1 = np.zeros((4, 4))
    dM1[0, 1] = dmx1_d1; dM1[1, 0] = dmx1_d1
    dM1[0, 2] = dmx2_d1; dM1[2, 0] = dmx2_d1
    dM1[0, 3] = dmx3_d1; dM1[3, 0] = dmx3_d1

    # k=2: ∂M/∂th2
    dM2 = np.zeros((4, 4))
    dM2[0, 1] = dmx1_d2; dM2[1, 0] = dmx1_d2
    dM2[0, 2] = dmx2_d2; dM2[2, 0] = dmx2_d2
    dM2[0, 3] = dmx3_d2; dM2[3, 0] = dmx3_d2
    dM2[1, 1] = dM11_d2
    dM2[1, 2] = dM12_d2; dM2[2, 1] = dM12_d2
    dM2[1, 3] = dM13_d2; dM2[3, 1] = dM13_d2

    # k=3: ∂M/∂th3
    dM3 = np.zeros((4, 4))
    dM3[0, 1] = dmx1_d3; dM3[1, 0] = dmx1_d3
    dM3[0, 2] = dmx2_d3; dM3[2, 0] = dmx2_d3
    dM3[0, 3] = dmx3_d3; dM3[3, 0] = dmx3_d3
    dM3[1, 1] = dM11_d3
    dM3[1, 2] = dM12_d3; dM3[2, 1] = dM12_d3
    dM3[1, 3] = dM13_d3; dM3[3, 1] = dM13_d3
    dM3[2, 2] = dM22_d3
    dM3[2, 3] = dM23_d3; dM3[3, 2] = dM23_d3

    dMs = (dM1, dM2, dM3)  # index 0→k=1, 1→k=2, 2→k=3

    for i in range(4):
        val = 0.0
        for j in range(4):
            for k in range(4):
                # dM_ij/dq_k
                if k == 0:
                    dMij_k = 0.0
                else:
                    dMij_k = dMs[k-1][i, j]
                # dM_ik/dq_j
                if j == 0:
                    dMik_j = 0.0
                else:
                    dMik_j = dMs[j-1][i, k]
                # dM_jk/dq_i
                if i == 0:
                    dMjk_i = 0.0
                else:
                    dMjk_i = dMs[i-1][j, k]

                c_ijk = 0.5 * (dMij_k + dMik_j - dMjk_i)
                val += c_ijk * dq[j] * dq[k]
        h[i] = val

    result = np.zeros((4, 1))
    result[0, 0] = h[0]
    result[1, 0] = h[1]
    result[2, 0] = h[2]
    result[3, 0] = h[3]
    return result
