"""Analytical Coriolis vector C(q,dq)*dq via closed-form partial derivatives of M."""

import numpy as np
from numba import njit

from dynamics.trigonometry import compute_trig


@njit(cache=True)
def coriolis_vector(q, dq, p):
    """Compute C(q,dq)*dq analytically. Returns (4,) flat array.

    Only nonzero Christoffel terms are evaluated (sparse expansion).
    """
    c1, c12, c123, c2, c23, c3, s1, s12, s123 = compute_trig(q)
    gx1, gx2, gx3 = p[1], p[2], p[3]
    b1, b2, b3 = p[7], p[8], p[9]

    dx, d1, d2, d3 = dq[0], dq[1], dq[2], dq[3]

    s2 = np.sin(q[2])
    s23_rel = np.sin(q[2] + q[3])
    s3 = np.sin(q[3])

    # Precompute dM/dq_k entries (k=1,2,3)
    # k=1 (dM1): only row/col 0 entries nonzero
    dmx1_d1 = -gx1*s1 - gx2*s12 - gx3*s123
    dmx2_d1 = -gx2*s12 - gx3*s123
    dmx3_d1 = -gx3*s123

    # k=2 (dM2): row/col 0 + pendulum block partial
    dmx1_d2 = -gx2*s12 - gx3*s123
    dmx2_d2 = -gx2*s12 - gx3*s123
    dmx3_d2 = -gx3*s123
    dM11_d2 = -2*b1*s2 - 2*b2*s23_rel
    dM12_d2 = -b1*s2 - b2*s23_rel
    dM13_d2 = -b2*s23_rel

    # k=3 (dM3): row/col 0 + pendulum block partial
    dmx1_d3 = -gx3*s123
    dmx2_d3 = -gx3*s123
    dmx3_d3 = -gx3*s123
    dM11_d3 = -2*b2*s23_rel - 2*b3*s3
    dM12_d3 = -b2*s23_rel - 2*b3*s3
    dM13_d3 = -b2*s23_rel - b3*s3
    dM22_d3 = -2*b3*s3
    dM23_d3 = -b3*s3

    # Christoffel symbol: c_ijk = 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
    # h[i] = sum_{j,k} c_ijk * dq[j] * dq[k]
    #
    # Key sparsity: dM/dq_0 = 0 always, so:
    #   - when k=0: dM_ij/dq_k=0
    #   - when j=0: dM_ik/dq_j=0
    #   - when i=0: dM_jk/dq_i=0
    # Also dM1 only has entries in row/col 0, so dM1[a,b]=0 if a>0 and b>0.

    # For clarity, define shorthand for dM_k[i,j] lookups:
    # dM1[0,1]=dM1[1,0]=dmx1_d1, dM1[0,2]=dM1[2,0]=dmx2_d1, dM1[0,3]=dM1[3,0]=dmx3_d1
    # dM2[0,1]=dM2[1,0]=dmx1_d2, dM2[0,2]=dM2[2,0]=dmx2_d2, dM2[0,3]=dM2[3,0]=dmx3_d2
    # dM2[1,1]=dM11_d2, dM2[1,2]=dM2[2,1]=dM12_d2, dM2[1,3]=dM2[3,1]=dM13_d2
    # dM3[0,1]=dM3[1,0]=dmx1_d3, dM3[0,2]=dM3[2,0]=dmx2_d3, dM3[0,3]=dM3[3,0]=dmx3_d3
    # dM3[1,1]=dM11_d3, dM3[1,2]=dM3[2,1]=dM12_d3, dM3[1,3]=dM3[3,1]=dM13_d3
    # dM3[2,2]=dM22_d3, dM3[2,3]=dM3[3,2]=dM23_d3

    # ---- h[0]: i=0, so dM_jk/dq_0 = 0 always ----
    # c_0jk = 0.5*(dM_0j/dq_k + dM_0k/dq_j)
    # dM_0j/dq_k nonzero only if j>0 and k>0 (since dM/dq_0=0)
    # dM_0j entries: dMk[0,j] for k=1,2,3 and j=1,2,3
    # So c_0jk = 0.5*(dMk[0,j] + dMj[0,k]) for j,k in {1,2,3}
    # h[0] = sum_{j=1..3, k=1..3} 0.5*(dMk[0,j] + dMj[0,k]) * dq[j]*dq[k]
    # By symmetry of j,k sum: h[0] = sum_{j,k=1..3} dMk[0,j] * dq[j]*dq[k]
    # (since swapping j,k in the second term gives the first)

    # Expand: for each k=1,2,3, sum_j dMk[0,j]*dq[j]*dq[k]
    # k=1: dM1[0,j] only nonzero for j=1,2,3 -> dmx1_d1*d1 + dmx2_d1*d2 + dmx3_d1*d3
    # k=2: dM2[0,j] -> dmx1_d2*d1 + dmx2_d2*d2 + dmx3_d2*d3
    # k=3: dM3[0,j] -> dmx1_d3*d1 + dmx2_d3*d2 + dmx3_d3*d3
    h0 = ((dmx1_d1*d1 + dmx2_d1*d2 + dmx3_d1*d3) * d1
         + (dmx1_d2*d1 + dmx2_d2*d2 + dmx3_d2*d3) * d2
         + (dmx1_d3*d1 + dmx2_d3*d2 + dmx3_d3*d3) * d3)

    # ---- h[1]: i=1 ----
    # c_1jk = 0.5*(dM_1j/dq_k + dM_1k/dq_j - dM_jk/dq_1)
    # dM_jk/dq_1 = dM1[j,k]: nonzero only when j=0 or k=0
    # dM_1j/dq_k: nonzero when k>0. For k>0, dMk[1,j].
    # dM_1k/dq_j: nonzero when j>0. For j>0, dMj[1,k].
    #
    # We need to handle j=0 and k=0 cases separately.

    # Let's systematically enumerate. Split into cases:
    # Case j=0, k=0: all three dM terms are 0 -> 0
    # Case j=0, k>0: dM_10/dq_k = dMk[1,0], dM_1k/dq_0 = 0, dM_0k/dq_1 = dM1[0,k]
    #   c = 0.5*(dMk[1,0] - dM1[0,k])  (note dMk[1,0] for k>0 and dM1[0,k] for k>0)
    #   But dMk[1,0] = dMk[0,1] (symmetric), and dM1[0,k] is the mx entries.
    #   For k=1: dM1[1,0]=dmx1_d1, dM1[0,1]=dmx1_d1 -> c=0
    #   For k=2: dM2[1,0]=dmx1_d2, dM1[0,2]=dmx2_d1 -> c=0.5*(dmx1_d2-dmx2_d1)
    #   For k=3: dM3[1,0]=dmx1_d3, dM1[0,3]=dmx3_d1 -> c=0.5*(dmx1_d3-dmx3_d1)
    # Case k=0, j>0: by symmetry with above (swap j,k labels)
    #   c_1j0 = 0.5*(dM_1j/dq_0 + dMj[1,0] - dM1[j,0])
    #         = 0.5*(0 + dMj[1,0] - dM1[j,0])
    #   For j=1: 0.5*(dmx1_d1-dmx1_d1)=0
    #   For j=2: 0.5*(dmx1_d2-dmx2_d1)
    #   For j=3: 0.5*(dmx1_d3-dmx3_d1)
    # Case j>0, k>0:
    #   dM_1j/dq_k = dMk[1,j], dM_1k/dq_j = dMj[1,k], dM_jk/dq_1 = dM1[j,k]
    #   dM1[j,k] for j,k>0: dM1 only has entries in row/col 0, so dM1[j,k]=0 for j,k>0.
    #   c = 0.5*(dMk[1,j] + dMj[1,k])

    # h[1] = sum over all j,k of c_1jk * dq[j]*dq[k]

    # j=0 terms: sum_k c_10k * dx * dq[k]
    # k=0: 0; k=1: 0; k=2: 0.5*(dmx1_d2-dmx2_d1)*dx*d2; k=3: 0.5*(dmx1_d3-dmx3_d1)*dx*d3
    t_j0 = 0.5*dx*(  (dmx1_d2 - dmx2_d1)*d2 + (dmx1_d3 - dmx3_d1)*d3 )

    # k=0 terms (j>0): sum_j c_1j0 * dq[j] * dx
    # j=1: 0; j=2: 0.5*(dmx1_d2-dmx2_d1)*d2*dx; j=3: 0.5*(dmx1_d3-dmx3_d1)*d3*dx
    t_k0 = 0.5*dx*(  (dmx1_d2 - dmx2_d1)*d2 + (dmx1_d3 - dmx3_d1)*d3 )

    # j>0, k>0 terms: c = 0.5*(dMk[1,j] + dMj[1,k])
    # Need dMk[1,j] for k,j in {1,2,3}:
    # dM1[1,j]: dM1[1,1]=0, dM1[1,2]=0, dM1[1,3]=0 (dM1 only has row/col 0 entries)
    # dM2[1,j]: dM2[1,1]=dM11_d2, dM2[1,2]=dM12_d2, dM2[1,3]=dM13_d2
    # dM3[1,j]: dM3[1,1]=dM11_d3, dM3[1,2]=dM12_d3, dM3[1,3]=dM13_d3

    # Enumerate (j,k) in {1,2,3}x{1,2,3}:
    # (1,1): 0.5*(dM1[1,1]+dM1[1,1])=0
    # (1,2): 0.5*(dM2[1,1]+dM1[1,2])=0.5*dM11_d2
    # (1,3): 0.5*(dM3[1,1]+dM1[1,3])=0.5*dM11_d3
    # (2,1): 0.5*(dM1[1,2]+dM2[1,1])=0.5*dM11_d2
    # (2,2): 0.5*(dM2[1,2]+dM2[1,2])=dM12_d2
    # (2,3): 0.5*(dM3[1,2]+dM2[1,3])=0.5*(dM12_d3+dM13_d2)
    # (3,1): 0.5*(dM1[1,3]+dM3[1,1])=0.5*dM11_d3
    # (3,2): 0.5*(dM2[1,3]+dM3[1,2])=0.5*(dM13_d2+dM12_d3)
    # (3,3): 0.5*(dM3[1,3]+dM3[1,3])=dM13_d3

    t_jk = (0.5*dM11_d2*(d1*d2 + d2*d1)
           + 0.5*dM11_d3*(d1*d3 + d3*d1)
           + dM12_d2*d2*d2
           + 0.5*(dM12_d3 + dM13_d2)*(d2*d3 + d3*d2)
           + dM13_d3*d3*d3)

    h1 = t_j0 + t_k0 + t_jk

    # ---- h[2]: i=2 ----
    # dM_jk/dq_2 = dM2[j,k]
    # c_2jk = 0.5*(dMk[2,j] + dMj[2,k] - dM2[j,k])

    # j=0, k=0: 0
    # j=0, k>0: 0.5*(dMk[2,0] + 0 - dM2[0,k])
    #   dMk[2,0]: dM1[2,0]=dmx2_d1, dM2[2,0]=dmx2_d2, dM3[2,0]=dmx2_d3
    #   dM2[0,k]: dM2[0,1]=dmx1_d2, dM2[0,2]=dmx2_d2, dM2[0,3]=dmx3_d2
    #   k=1: 0.5*(dmx2_d1-dmx1_d2)*dx*d1
    #   k=2: 0.5*(dmx2_d2-dmx2_d2)=0
    #   k=3: 0.5*(dmx2_d3-dmx3_d2)*dx*d3
    # k=0, j>0: 0.5*(0 + dMj[2,0] - dM2[j,0])
    #   dMj[2,0]: dM1[2,0]=dmx2_d1, dM2[2,0]=dmx2_d2, dM3[2,0]=dmx2_d3
    #   dM2[j,0]: dM2[1,0]=dmx1_d2, dM2[2,0]=dmx2_d2, dM2[3,0]=dmx3_d2
    #   j=1: 0.5*(dmx2_d1-dmx1_d2)*d1*dx
    #   j=2: 0
    #   j=3: 0.5*(dmx2_d3-dmx3_d2)*d3*dx

    u_j0k0 = dx * ((dmx2_d1 - dmx1_d2)*d1 + (dmx2_d3 - dmx3_d2)*d3)

    # j>0, k>0: 0.5*(dMk[2,j] + dMj[2,k] - dM2[j,k])
    # Need dMk[2,j], dMj[2,k], dM2[j,k] for j,k in {1,2,3}
    # dM1[2,j]: dM1[2,0..3] = [dmx2_d1,0,0,0] -> for j>0: 0
    # dM2[2,j]: j=1: dM12_d2, j=2: 0, j=3: 0
    # dM3[2,j]: j=1: dM12_d3, j=2: dM22_d3, j=3: dM23_d3

    # dMj[2,k] = same structure swapping j for the derivative index
    # dM2[j,k] for j,k>0:
    #   (1,1)=dM11_d2, (1,2)=dM12_d2, (1,3)=dM13_d2
    #   (2,1)=dM12_d2, (2,2)=0, (2,3)=0
    #   (3,1)=dM13_d2, (3,2)=0, (3,3)=0

    # Enumerate (j,k):
    # (1,1): 0.5*(dM1[2,1]+dM1[2,1]-dM2[1,1])=0.5*(0+0-dM11_d2)=-0.5*dM11_d2
    # (1,2): 0.5*(dM2[2,1]+dM1[2,2]-dM2[1,2])=0.5*(dM12_d2+0-dM12_d2)=0
    # (1,3): 0.5*(dM3[2,1]+dM1[2,3]-dM2[1,3])=0.5*(dM12_d3+0-dM13_d2)=0.5*(dM12_d3-dM13_d2)
    # (2,1): 0.5*(dM1[2,2]+dM2[2,1]-dM2[2,1])=0.5*(0+dM12_d2-dM12_d2)=0
    # (2,2): 0.5*(dM2[2,2]+dM2[2,2]-dM2[2,2])=0.5*0=0
    # (2,3): 0.5*(dM3[2,2]+dM2[2,3]-dM2[2,3])=0.5*(dM22_d3+0-0)=0.5*dM22_d3
    # (3,1): 0.5*(dM1[2,3]+dM3[2,1]-dM2[3,1])=0.5*(0+dM12_d3-dM13_d2)=0.5*(dM12_d3-dM13_d2)
    # (3,2): 0.5*(dM2[2,3]+dM3[2,2]-dM2[3,2])=0.5*(0+dM22_d3-0)=0.5*dM22_d3
    # (3,3): 0.5*(dM3[2,3]+dM3[2,3]-dM2[3,3])=0.5*(dM23_d3+dM23_d3-0)=dM23_d3

    u_jk = (-0.5*dM11_d2*d1*d1
            + 0.5*(dM12_d3 - dM13_d2)*(d1*d3 + d3*d1)
            + 0.5*dM22_d3*(d2*d3 + d3*d2)
            + dM23_d3*d3*d3)

    h2 = u_j0k0 + u_jk

    # ---- h[3]: i=3 ----
    # dM_jk/dq_3 = dM3[j,k]
    # c_3jk = 0.5*(dMk[3,j] + dMj[3,k] - dM3[j,k])

    # j=0, k=0: 0
    # j=0, k>0: 0.5*(dMk[3,0] - dM3[0,k])
    #   dMk[3,0]: dM1[3,0]=dmx3_d1, dM2[3,0]=dmx3_d2, dM3[3,0]=dmx3_d3
    #   dM3[0,k]: dM3[0,1]=dmx1_d3, dM3[0,2]=dmx2_d3, dM3[0,3]=dmx3_d3
    #   k=1: 0.5*(dmx3_d1-dmx1_d3)
    #   k=2: 0.5*(dmx3_d2-dmx2_d3)
    #   k=3: 0.5*(dmx3_d3-dmx3_d3)=0
    # k=0, j>0: 0.5*(dMj[3,0] - dM3[j,0])
    #   same values by symmetry
    #   j=1: 0.5*(dmx3_d1-dmx1_d3)
    #   j=2: 0.5*(dmx3_d2-dmx2_d3)
    #   j=3: 0

    v_j0k0 = dx * ((dmx3_d1 - dmx1_d3)*d1 + (dmx3_d2 - dmx2_d3)*d2)

    # j>0, k>0: 0.5*(dMk[3,j] + dMj[3,k] - dM3[j,k])
    # dMk[3,j] for k,j in {1,2,3}:
    # dM1[3,j]: j=1..3 -> all 0 (dM1 only row/col 0)
    # dM2[3,j]: j=1: dM13_d2, j=2: 0, j=3: 0
    # dM3[3,j]: j=1: dM13_d3, j=2: dM23_d3, j=3: 0

    # dM3[j,k] for j,k>0:
    #   (1,1)=dM11_d3, (1,2)=dM12_d3, (1,3)=dM13_d3
    #   (2,1)=dM12_d3, (2,2)=dM22_d3, (2,3)=dM23_d3
    #   (3,1)=dM13_d3, (3,2)=dM23_d3, (3,3)=0

    # Enumerate:
    # (1,1): 0.5*(dM1[3,1]+dM1[3,1]-dM3[1,1])=0.5*(0+0-dM11_d3)=-0.5*dM11_d3
    # (1,2): 0.5*(dM2[3,1]+dM1[3,2]-dM3[1,2])=0.5*(dM13_d2+0-dM12_d3)=0.5*(dM13_d2-dM12_d3)
    # (1,3): 0.5*(dM3[3,1]+dM1[3,3]-dM3[1,3])=0.5*(dM13_d3+0-dM13_d3)=0
    # (2,1): 0.5*(dM1[3,2]+dM2[3,1]-dM3[2,1])=0.5*(0+dM13_d2-dM12_d3)=0.5*(dM13_d2-dM12_d3)
    # (2,2): 0.5*(dM2[3,2]+dM2[3,2]-dM3[2,2])=0.5*(0+0-dM22_d3)=-0.5*dM22_d3
    # (2,3): 0.5*(dM3[3,2]+dM2[3,3]-dM3[2,3])=0.5*(dM23_d3+0-dM23_d3)=0
    # (3,1): 0.5*(dM1[3,3]+dM3[3,1]-dM3[3,1])=0.5*(0+dM13_d3-dM13_d3)=0
    # (3,2): 0.5*(dM2[3,3]+dM3[3,2]-dM3[3,2])=0.5*(0+dM23_d3-dM23_d3)=0
    # (3,3): 0.5*(dM3[3,3]+dM3[3,3]-dM3[3,3])=0.5*(0+0-0)=0

    v_jk = (-0.5*dM11_d3*d1*d1
            + 0.5*(dM13_d2 - dM12_d3)*(d1*d2 + d2*d1)
            - 0.5*dM22_d3*d2*d2)

    h3 = v_j0k0 + v_jk

    h = np.empty(4)
    h[0] = h0
    h[1] = h1
    h[2] = h2
    h[3] = h3
    return h
