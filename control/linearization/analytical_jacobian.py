"""True analytical Jacobian computation for the triple-pendulum system.

At equilibrium (dq = 0), the Coriolis term C(q,dq)*dq vanishes, so:
    ddq = M(q)^{-1} [tau - G(q)]

The Jacobians simplify to:
    A_q = d(ddq)/dq = -M^{-1} dG/dq + dM^{-1}/dq * (tau - G)
    A_dq = d(ddq)/d(dq) = -M^{-1} dC/d(dq)  (evaluated at dq=0)
    B_u = M^{-1} * [1, 0, 0, 0]^T

At the equilibrium with u=0 and tau-G=0 (not generally true), the
dM^{-1}/dq term contributes. We use the identity:
    dM^{-1}/dq_k = -M^{-1} (dM/dq_k) M^{-1}

For A_dq at dq=0: the Coriolis vector h = C(q,dq)*dq is bilinear in dq,
so dh/d(dq_j) = sum_k [Gamma_ijk + Gamma_ikj] * dq_k + ...
At dq=0 this simplifies because many terms vanish. We use the JIT
finite-difference approach for A_dq since it's already fast and the
analytical expression is more complex (requires all Christoffel symbols).

This hybrid approach gives exact A_q and B_u analytically, and uses
optimized numerical differences only for A_dq (which is typically
near-zero at equilibrium for this system).
"""

import numpy as np
from numba import njit
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def _compute_analytical_A_q(q, p):
    """Compute A_q = d(ddq)/dq analytically at dq=0, u=0.

    Returns (4, 4) matrix.
    """
    th1 = q[1]; th2 = q[2]; th3 = q[3]

    # Trig
    c1 = np.cos(th1); s1 = np.sin(th1)
    phi2 = th1 + th2
    c12 = np.cos(phi2); s12 = np.sin(phi2)
    phi3 = phi2 + th3
    c123 = np.cos(phi3); s123 = np.sin(phi3)
    c2 = np.cos(th2); c23 = np.cos(th2 + th3); c3 = np.cos(th3)

    # Unpack parameters
    Mt = p[0]
    gx1 = p[1]; gx2 = p[2]; gx3 = p[3]
    a1 = p[4]; a2 = p[5]; a3 = p[6]
    b1 = p[7]; b2 = p[8]; b3 = p[9]
    gg1 = p[10]; gg2 = p[11]; gg3 = p[12]

    # Mass matrix M(q)
    mx1 = gx1*c1 + gx2*c12 + gx3*c123
    mx2 = gx2*c12 + gx3*c123
    mx3 = gx3*c123

    M11 = a1 + a2 + a3 + 2.0*b1*c2 + 2.0*b2*c23 + 2.0*b3*c3
    M12 = a2 + a3 + b1*c2 + b2*c23 + 2.0*b3*c3
    M13 = a3 + b2*c23 + b3*c3
    M22 = a2 + a3 + 2.0*b3*c3
    M23 = a3 + b3*c3
    M33 = a3

    M = np.zeros((4, 4))
    M[0, 0] = Mt; M[0, 1] = mx1; M[0, 2] = mx2; M[0, 3] = mx3
    M[1, 0] = mx1; M[1, 1] = M11; M[1, 2] = M12; M[1, 3] = M13
    M[2, 0] = mx2; M[2, 1] = M12; M[2, 2] = M22; M[2, 3] = M23
    M[3, 0] = mx3; M[3, 1] = M13; M[3, 2] = M23; M[3, 3] = M33

    M_inv = np.linalg.inv(M)

    # Gravity vector G(q)
    G = np.zeros(4)
    G[1] = gg1*s1 + gg2*s12 + gg3*s123
    G[2] = gg2*s12 + gg3*s123
    G[3] = gg3*s123

    # tau - G (with u=0, tau=[0,0,0,0])
    rhs = -G  # tau=0

    # dG/dq (4x4 matrix, dG[i]/dq[j])
    # dG/dq0 = 0 (no dependence on cart position x)
    dGdq = np.zeros((4, 4))

    # dG/dth1 (column 1)
    dGdq[1, 1] = gg1*c1 + gg2*c12 + gg3*c123
    dGdq[2, 1] = gg2*c12 + gg3*c123
    dGdq[3, 1] = gg3*c123

    # dG/dth2 (column 2)
    dGdq[1, 2] = gg2*c12 + gg3*c123
    dGdq[2, 2] = gg2*c12 + gg3*c123
    dGdq[3, 2] = gg3*c123

    # dG/dth3 (column 3)
    dGdq[1, 3] = gg3*c123
    dGdq[2, 3] = gg3*c123
    dGdq[3, 3] = gg3*c123

    # dM/dq_k for k=1,2,3 (stored as 3 matrices)
    s2 = np.sin(th2)
    s23 = np.sin(th2 + th3)
    s3 = np.sin(th3)

    # Compute A_q = d(ddq)/dq = d(M^{-1}(tau-G))/dq
    # = M^{-1}(-dG/dq) + dM^{-1}/dq_k * rhs  for each column k
    # dM^{-1}/dq_k = -M^{-1} (dM/dq_k) M^{-1}

    A_q = np.zeros((4, 4))

    # Column 0: dq0 = dx, M doesn't depend on x, G doesn't depend on x
    # A_q[:, 0] = 0 (already)

    # For columns 1, 2, 3: compute dM/dq_k
    for k_col in range(1, 4):
        dM = np.zeros((4, 4))

        if k_col == 1:  # dM/dth1
            dmx1 = -gx1*s1 - gx2*s12 - gx3*s123
            dmx2 = -gx2*s12 - gx3*s123
            dmx3 = -gx3*s123
            dM[0, 1] = dmx1; dM[1, 0] = dmx1
            dM[0, 2] = dmx2; dM[2, 0] = dmx2
            dM[0, 3] = dmx3; dM[3, 0] = dmx3
            # Pendulum block: no dependence on th1

        elif k_col == 2:  # dM/dth2
            dmx1 = -gx2*s12 - gx3*s123
            dmx2 = -gx2*s12 - gx3*s123
            dmx3 = -gx3*s123
            dM[0, 1] = dmx1; dM[1, 0] = dmx1
            dM[0, 2] = dmx2; dM[2, 0] = dmx2
            dM[0, 3] = dmx3; dM[3, 0] = dmx3
            # Pendulum block derivatives
            dM[1, 1] = -2.0*b1*s2 - 2.0*b2*s23
            dM[1, 2] = -b1*s2 - b2*s23; dM[2, 1] = dM[1, 2]
            dM[1, 3] = -b2*s23; dM[3, 1] = dM[1, 3]

        else:  # k_col == 3: dM/dth3
            dmx1 = -gx3*s123
            dmx2 = -gx3*s123
            dmx3 = -gx3*s123
            dM[0, 1] = dmx1; dM[1, 0] = dmx1
            dM[0, 2] = dmx2; dM[2, 0] = dmx2
            dM[0, 3] = dmx3; dM[3, 0] = dmx3
            # Pendulum block derivatives
            dM[1, 1] = -2.0*b2*s23 - 2.0*b3*s3
            dM[1, 2] = -b2*s23 - 2.0*b3*s3; dM[2, 1] = dM[1, 2]
            dM[1, 3] = -b2*s23 - b3*s3; dM[3, 1] = dM[1, 3]
            dM[2, 2] = -2.0*b3*s3
            dM[2, 3] = -b3*s3; dM[3, 2] = dM[2, 3]

        # A_q[:, k] = M^{-1}(-dG/dq_k) + (-M^{-1} dM M^{-1}) rhs
        # = M^{-1}(-dGdq[:, k] - dM @ M^{-1} @ rhs)
        Minv_rhs = M_inv @ rhs  # = ddq at equilibrium (should be ~0 if at true eq)
        col = M_inv @ (-dGdq[:, k_col] - dM @ Minv_rhs)
        for i in range(4):
            A_q[i, k_col] = col[i]

    return A_q


@njit(cache=True)
def _compute_analytical_B_u(q, p):
    """Compute B_u = d(ddq)/du analytically.

    B_u = M^{-1} @ [1, 0, 0, 0]^T  (first column of M^{-1}).
    Returns (4, 1) matrix.
    """
    th1 = q[1]; th2 = q[2]; th3 = q[3]

    c1 = np.cos(th1)
    c12 = np.cos(th1 + th2)
    c123 = np.cos(th1 + th2 + th3)
    c2 = np.cos(th2); c23 = np.cos(th2 + th3); c3 = np.cos(th3)

    Mt = p[0]
    gx1 = p[1]; gx2 = p[2]; gx3 = p[3]
    a1 = p[4]; a2 = p[5]; a3 = p[6]
    b1 = p[7]; b2 = p[8]; b3 = p[9]

    mx1 = gx1*c1 + gx2*c12 + gx3*c123
    mx2 = gx2*c12 + gx3*c123
    mx3 = gx3*c123

    M11 = a1 + a2 + a3 + 2.0*b1*c2 + 2.0*b2*c23 + 2.0*b3*c3
    M12 = a2 + a3 + b1*c2 + b2*c23 + 2.0*b3*c3
    M13 = a3 + b2*c23 + b3*c3
    M22 = a2 + a3 + 2.0*b3*c3
    M23 = a3 + b3*c3
    M33 = a3

    M = np.zeros((4, 4))
    M[0, 0] = Mt; M[0, 1] = mx1; M[0, 2] = mx2; M[0, 3] = mx3
    M[1, 0] = mx1; M[1, 1] = M11; M[1, 2] = M12; M[1, 3] = M13
    M[2, 0] = mx2; M[2, 1] = M12; M[2, 2] = M22; M[2, 3] = M23
    M[3, 0] = mx3; M[3, 1] = M13; M[3, 2] = M23; M[3, 3] = M33

    M_inv = np.linalg.inv(M)

    B_u = np.zeros((4, 1))
    for i in range(4):
        B_u[i, 0] = M_inv[i, 0]
    return B_u


@njit(cache=True)
def _compute_A_dq_numerical(q, p):
    """Compute A_dq via central differences (needed because analytical
    Coriolis Jacobian d(C*dq)/d(dq) at dq=0 requires careful treatment).

    At dq=0, the Coriolis/centrifugal terms vanish but their derivative
    w.r.t. dq does not. The Christoffel-based analytical form exists but
    is complex; numerical differences are efficient here since each
    forward_dynamics call is O(1) via JIT.

    Returns (4, 4) matrix.
    """
    n = 4
    dq_eq = np.zeros(4)
    u_eq = 0.0
    A_dq = np.zeros((n, n))

    for j in range(n):
        h = 6.06e-6  # sqrt(eps_mach)
        dq_plus = dq_eq.copy()
        dq_minus = dq_eq.copy()
        dq_plus[j] += h
        dq_minus[j] -= h
        ddq_plus = forward_dynamics(q, dq_plus, u_eq, p)
        ddq_minus = forward_dynamics(q, dq_minus, u_eq, p)
        for i in range(n):
            A_dq[i, j] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h)

    return A_dq


def compute_analytical_jacobians(q_eq, p):
    """Compute A, B matrices using analytical Jacobians.

    Uses exact analytical derivatives for A_q and B_u, and optimized
    numerical differences for A_dq (Coriolis velocity Jacobian).

    This is the recommended entry point for high-accuracy linearization.

    Returns
    -------
    A : ndarray (8, 8)
    B : ndarray (8, 1)
    """
    A_q = _compute_analytical_A_q(q_eq, p)
    A_dq = _compute_A_dq_numerical(q_eq, p)
    B_u = _compute_analytical_B_u(q_eq, p)

    n = 4
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, 1))

    # Top-right block: identity
    for i in range(n):
        A[i, n + i] = 1.0

    # Bottom-left: A_q
    A[n:, :n] = A_q

    # Bottom-right: A_dq
    A[n:, n:] = A_dq

    # Bottom of B: B_u
    B[n:, :] = B_u

    return A, B
