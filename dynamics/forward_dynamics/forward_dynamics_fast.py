"""Zero-allocation monolithic forward dynamics.

Computes ddq = M^{-1}(tau - C - G) with:
- Single trig computation (9 calls instead of 27)
- No intermediate array allocations
- Inline cofactor 4x4 solve
- All scalar operations
"""

import numpy as np
from numba import njit, float64

_F64 = float64

@njit(float64(float64, float64, float64, float64, float64, float64, float64, float64, float64),
      cache=True, fastmath=True, boundscheck=False)
def _det3(a, b, c, d, e, f, g, h, i):
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


@njit(cache=True, fastmath=True, boundscheck=False)
def forward_dynamics_fast(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p):
    """Compute 4 accelerations from scalar state. Returns (ddq0, ddq1, ddq2, ddq3).

    All operations are scalar — zero heap allocation.
    """
    # ---- Trig (computed ONCE) ----
    c1 = np.cos(q1)
    s1 = np.sin(q1)
    phi2 = q1 + q2
    c12 = np.cos(phi2)
    s12 = np.sin(phi2)
    phi3 = phi2 + q3
    c123 = np.cos(phi3)
    s123 = np.sin(phi3)
    c2 = np.cos(q2)
    s2 = np.sin(q2)
    th23 = q2 + q3
    c23 = np.cos(th23)
    s23 = np.sin(th23)
    c3 = np.cos(q3)
    s3 = np.sin(q3)

    # ---- Unpack parameters ----
    Mt = p[0]
    gx1 = p[1]; gx2 = p[2]; gx3 = p[3]
    a1 = p[4]; a2 = p[5]; a3 = p[6]
    b1 = p[7]; b2 = p[8]; b3 = p[9]
    gg1 = p[10]; gg2 = p[11]; gg3 = p[12]

    # ---- Mass matrix (16 scalars) ----
    mx1 = gx1 * c1 + gx2 * c12 + gx3 * c123
    mx2 = gx2 * c12 + gx3 * c123
    mx3 = gx3 * c123

    M11 = a1 + a2 + a3 + 2.0 * b1 * c2 + 2.0 * b2 * c23 + 2.0 * b3 * c3
    M12 = a2 + a3 + b1 * c2 + b2 * c23 + 2.0 * b3 * c3
    M13 = a3 + b2 * c23 + b3 * c3
    M22 = a2 + a3 + 2.0 * b3 * c3
    M23 = a3 + b3 * c3
    M33 = a3

    m00 = Mt;  m01 = mx1; m02 = mx2; m03 = mx3
    m10 = mx1; m11 = M11; m12 = M12; m13 = M13
    m20 = mx2; m21 = M12; m22 = M22; m23 = M23
    m30 = mx3; m31 = M13; m32 = M23; m33 = M33

    # ---- Coriolis vector (scalar) ----
    dmx1_d1 = -gx1 * s1 - gx2 * s12 - gx3 * s123
    dmx2_d1 = -gx2 * s12 - gx3 * s123
    dmx3_d1 = -gx3 * s123

    dmx1_d2 = -gx2 * s12 - gx3 * s123
    dmx2_d2 = -gx2 * s12 - gx3 * s123
    dmx3_d2 = -gx3 * s123
    dM11_d2 = -2.0 * b1 * s2 - 2.0 * b2 * s23
    dM12_d2 = -b1 * s2 - b2 * s23
    dM13_d2 = -b2 * s23

    dmx1_d3 = -gx3 * s123
    dmx2_d3 = -gx3 * s123
    dmx3_d3 = -gx3 * s123
    dM11_d3 = -2.0 * b2 * s23 - 2.0 * b3 * s3
    dM12_d3 = -b2 * s23 - 2.0 * b3 * s3
    dM13_d3 = -b2 * s23 - b3 * s3
    dM22_d3 = -2.0 * b3 * s3
    dM23_d3 = -b3 * s3

    dx = dq0; d1 = dq1; d2 = dq2; d3 = dq3

    h0 = ((dmx1_d1 * d1 + dmx2_d1 * d2 + dmx3_d1 * d3) * d1
         + (dmx1_d2 * d1 + dmx2_d2 * d2 + dmx3_d2 * d3) * d2
         + (dmx1_d3 * d1 + dmx2_d3 * d2 + dmx3_d3 * d3) * d3)

    t_j0 = 0.5 * dx * ((dmx1_d2 - dmx2_d1) * d2 + (dmx1_d3 - dmx3_d1) * d3)
    t_k0 = t_j0  # symmetric
    t_jk = (dM11_d2 * d1 * d2
           + dM11_d3 * d1 * d3
           + dM12_d2 * d2 * d2
           + (dM12_d3 + dM13_d2) * d2 * d3
           + dM13_d3 * d3 * d3)
    h1 = t_j0 + t_k0 + t_jk

    u_j0k0 = dx * ((dmx2_d1 - dmx1_d2) * d1 + (dmx2_d3 - dmx3_d2) * d3)
    u_jk = (-0.5 * dM11_d2 * d1 * d1
            + (dM12_d3 - dM13_d2) * d1 * d3
            + dM22_d3 * d2 * d3
            + dM23_d3 * d3 * d3)
    h2 = u_j0k0 + u_jk

    v_j0k0 = dx * ((dmx3_d1 - dmx1_d3) * d1 + (dmx3_d2 - dmx2_d3) * d2)
    v_jk = (-0.5 * dM11_d3 * d1 * d1
            + (dM13_d2 - dM12_d3) * d1 * d2
            - 0.5 * dM22_d3 * d2 * d2)
    h3 = v_j0k0 + v_jk

    # ---- Gravity (scalar, sin only) ----
    G1 = gg1 * s1 + gg2 * s12 + gg3 * s123
    G2 = gg2 * s12 + gg3 * s123
    G3 = gg3 * s123

    # ---- RHS = tau - C - G (scalar) ----
    r0 = u - h0
    r1 = -h1 - G1
    r2 = -h2 - G2
    r3 = -h3 - G3

    # ---- Solve M * ddq = rhs via cofactor expansion (inline) ----
    A00 = _det3(m11, m12, m13, m21, m22, m23, m31, m32, m33)
    A01 = -_det3(m10, m12, m13, m20, m22, m23, m30, m32, m33)
    A02 = _det3(m10, m11, m13, m20, m21, m23, m30, m31, m33)
    A03 = -_det3(m10, m11, m12, m20, m21, m22, m30, m31, m32)

    det = m00 * A00 + m01 * A01 + m02 * A02 + m03 * A03

    A10 = -_det3(m01, m02, m03, m21, m22, m23, m31, m32, m33)
    A11 = _det3(m00, m02, m03, m20, m22, m23, m30, m32, m33)
    A12 = -_det3(m00, m01, m03, m20, m21, m23, m30, m31, m33)
    A13 = _det3(m00, m01, m02, m20, m21, m22, m30, m31, m32)

    A20 = _det3(m01, m02, m03, m11, m12, m13, m31, m32, m33)
    A21 = -_det3(m00, m02, m03, m10, m12, m13, m30, m32, m33)
    A22 = _det3(m00, m01, m03, m10, m11, m13, m30, m31, m33)
    A23 = -_det3(m00, m01, m02, m10, m11, m12, m30, m31, m32)

    A30 = -_det3(m01, m02, m03, m11, m12, m13, m21, m22, m23)
    A31 = _det3(m00, m02, m03, m10, m12, m13, m20, m22, m23)
    A32 = -_det3(m00, m01, m03, m10, m11, m13, m20, m21, m23)
    A33 = _det3(m00, m01, m02, m10, m11, m12, m20, m21, m22)

    diag_max = max(abs(m00), abs(m11), abs(m22), abs(m33))
    if abs(det) < 1e-12 * (diag_max * diag_max * diag_max * diag_max + 1e-30):
        return 0.0, 0.0, 0.0, 0.0

    inv_det = 1.0 / det

    ddq0 = (A00 * r0 + A10 * r1 + A20 * r2 + A30 * r3) * inv_det
    ddq1 = (A01 * r0 + A11 * r1 + A21 * r2 + A31 * r3) * inv_det
    ddq2 = (A02 * r0 + A12 * r1 + A22 * r2 + A32 * r3) * inv_det
    ddq3 = (A03 * r0 + A13 * r1 + A23 * r2 + A33 * r3) * inv_det

    return ddq0, ddq1, ddq2, ddq3


@njit(cache=True, fastmath=True, boundscheck=False)
def rk4_step_fast(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p, dt):
    """Single RK4 step using scalar state. Zero allocation."""
    hdt = 0.5 * dt
    s6 = dt / 6.0

    # k1
    a0, a1, a2, a3 = forward_dynamics_fast(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p)
    # k1 for positions = dq, k1 for velocities = a
    k1_q0 = dq0;  k1_q1 = dq1;  k1_q2 = dq2;  k1_q3 = dq3
    k1_dq0 = a0;  k1_dq1 = a1;  k1_dq2 = a2;  k1_dq3 = a3

    # k2
    tq0 = q0 + hdt * k1_q0;   tq1 = q1 + hdt * k1_q1
    tq2 = q2 + hdt * k1_q2;   tq3 = q3 + hdt * k1_q3
    tdq0 = dq0 + hdt * k1_dq0; tdq1 = dq1 + hdt * k1_dq1
    tdq2 = dq2 + hdt * k1_dq2; tdq3 = dq3 + hdt * k1_dq3
    a0, a1, a2, a3 = forward_dynamics_fast(tq0, tq1, tq2, tq3, tdq0, tdq1, tdq2, tdq3, u, p)
    k2_q0 = tdq0;  k2_q1 = tdq1;  k2_q2 = tdq2;  k2_q3 = tdq3
    k2_dq0 = a0;   k2_dq1 = a1;   k2_dq2 = a2;   k2_dq3 = a3

    # k3
    tq0 = q0 + hdt * k2_q0;   tq1 = q1 + hdt * k2_q1
    tq2 = q2 + hdt * k2_q2;   tq3 = q3 + hdt * k2_q3
    tdq0 = dq0 + hdt * k2_dq0; tdq1 = dq1 + hdt * k2_dq1
    tdq2 = dq2 + hdt * k2_dq2; tdq3 = dq3 + hdt * k2_dq3
    a0, a1, a2, a3 = forward_dynamics_fast(tq0, tq1, tq2, tq3, tdq0, tdq1, tdq2, tdq3, u, p)
    k3_q0 = tdq0;  k3_q1 = tdq1;  k3_q2 = tdq2;  k3_q3 = tdq3
    k3_dq0 = a0;   k3_dq1 = a1;   k3_dq2 = a2;   k3_dq3 = a3

    # k4
    tq0 = q0 + dt * k3_q0;   tq1 = q1 + dt * k3_q1
    tq2 = q2 + dt * k3_q2;   tq3 = q3 + dt * k3_q3
    tdq0 = dq0 + dt * k3_dq0; tdq1 = dq1 + dt * k3_dq1
    tdq2 = dq2 + dt * k3_dq2; tdq3 = dq3 + dt * k3_dq3
    a0, a1, a2, a3 = forward_dynamics_fast(tq0, tq1, tq2, tq3, tdq0, tdq1, tdq2, tdq3, u, p)
    k4_q0 = tdq0;  k4_q1 = tdq1;  k4_q2 = tdq2;  k4_q3 = tdq3
    k4_dq0 = a0;   k4_dq1 = a1;   k4_dq2 = a2;   k4_dq3 = a3

    # Combine
    q0_new = q0 + s6 * (k1_q0 + 2.0 * k2_q0 + 2.0 * k3_q0 + k4_q0)
    q1_new = q1 + s6 * (k1_q1 + 2.0 * k2_q1 + 2.0 * k3_q1 + k4_q1)
    q2_new = q2 + s6 * (k1_q2 + 2.0 * k2_q2 + 2.0 * k3_q2 + k4_q2)
    q3_new = q3 + s6 * (k1_q3 + 2.0 * k2_q3 + 2.0 * k3_q3 + k4_q3)
    dq0_new = dq0 + s6 * (k1_dq0 + 2.0 * k2_dq0 + 2.0 * k3_dq0 + k4_dq0)
    dq1_new = dq1 + s6 * (k1_dq1 + 2.0 * k2_dq1 + 2.0 * k3_dq1 + k4_dq1)
    dq2_new = dq2 + s6 * (k1_dq2 + 2.0 * k2_dq2 + 2.0 * k3_dq2 + k4_dq2)
    dq3_new = dq3 + s6 * (k1_dq3 + 2.0 * k2_dq3 + 2.0 * k3_dq3 + k4_dq3)

    return q0_new, q1_new, q2_new, q3_new, dq0_new, dq1_new, dq2_new, dq3_new
