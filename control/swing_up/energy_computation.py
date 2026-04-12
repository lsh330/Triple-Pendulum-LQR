"""Scalar JIT computation of total mechanical energy.

PE sign convention
------------------
  PE_code = gg1*cos(phi1) + gg2*cos(phi2) + gg3*cos(phi3)
  V_physical = -PE_code
  Physical total energy: E = KE - PE_code = KE + V_physical

  At equilibrium (KE=0):
    E* = -PE_code_eq = -(gg1*cos(phi1*) + gg2*cos(phi2*) + gg3*cos(phi3*))

Mass matrix elements mirror dynamics/mass_matrix/ exactly:
  cart_link_coupling: mx_i = gx_i * cos(phi_i)  (absolute angles)
  pendulum_block uses RELATIVE angles (theta2, theta3):
    c2   = cos(theta2)
    c23  = cos(theta2 + theta3)
    c3   = cos(theta3)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def total_energy_scalar(sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, p):
    """Compute physical total energy E = KE - PE_code.

    All scalar operations, zero allocation. Matches mass_matrix/assembly.py
    and gravity/gravity_vector.py exactly.

    Parameters
    ----------
    sq0..sq3 : float
        Configuration scalars [x, theta1, theta2, theta3].
    sdq0..sdq3 : float
        Velocity scalars [dx, dtheta1, dtheta2, dtheta3].
    p : 1-D array (13,)
        Packed parameter vector (see parameters/packing.py).

    Returns
    -------
    float
        Physical total energy E (J).
    """
    # Absolute angles
    phi1 = sq1
    phi2 = sq1 + sq2
    phi3 = sq1 + sq2 + sq3

    c1   = np.cos(phi1)
    c2   = np.cos(phi2)
    c3   = np.cos(phi3)

    # Relative-angle trig (used in pendulum block — matches trigonometry.py)
    c_th2   = np.cos(sq2)           # cos(theta2)
    c_th3   = np.cos(sq3)           # cos(theta3)
    c_th23  = np.cos(sq2 + sq3)     # cos(theta2 + theta3)

    Mt  = p[0]
    gx1 = p[1];  gx2 = p[2];  gx3 = p[3]
    a1  = p[4];  a2  = p[5];  a3  = p[6]
    b1  = p[7];  b2  = p[8];  b3  = p[9]
    gg1 = p[10]; gg2 = p[11]; gg3 = p[12]

    # Cart-link coupling (cart_link_coupling.py)
    mx1 = gx1 * c1 + gx2 * c2 + gx3 * c3
    mx2 = gx2 * c2 + gx3 * c3
    mx3 = gx3 * c3

    # Pendulum block (pendulum_block.py)
    M11 = a1 + a2 + a3 + 2.0*b1*c_th2 + 2.0*b2*c_th23 + 2.0*b3*c_th3
    M12 = a2 + a3 + b1*c_th2 + b2*c_th23 + 2.0*b3*c_th3
    M13 = a3 + b2*c_th23 + b3*c_th3
    M22 = a2 + a3 + 2.0*b3*c_th3
    M23 = a3 + b3*c_th3
    M33 = a3

    # KE = 0.5 * dq^T M dq  (symmetric 4x4 expansion, rows 0..3)
    KE = 0.5 * (
        Mt    * sdq0 * sdq0
        + 2.0 * mx1 * sdq0 * sdq1
        + 2.0 * mx2 * sdq0 * sdq2
        + 2.0 * mx3 * sdq0 * sdq3
        + M11 * sdq1 * sdq1
        + 2.0 * M12 * sdq1 * sdq2
        + 2.0 * M13 * sdq1 * sdq3
        + M22 * sdq2 * sdq2
        + 2.0 * M23 * sdq2 * sdq3
        + M33 * sdq3 * sdq3
    )

    # PE_code = gg1*cos(phi1) + gg2*cos(phi2) + gg3*cos(phi3)
    # Derivation: PE = sum m_i*g*h_i with cumulative heights expands to
    # (m1*L1c + m2*L1 + m3*L1)*g*cos(phi1) + ... = gx_i*g*cos(phi_i) = gg_i*cos(phi_i)
    PE_code = gg1 * c1 + gg2 * c2 + gg3 * c3

    # Physical energy: E = KE + V_physical = KE - PE_code
    return KE - PE_code


@njit(cache=True)
def target_energy_from_phis(phi1_target, phi2_target, phi3_target, p):
    """Compute target physical energy at an equilibrium (KE=0).

    E* = V_physical = -PE_code = -(gg1*cos(phi1*) + gg2*cos(phi2*) + gg3*cos(phi3*))

    Parameters
    ----------
    phi1_target, phi2_target, phi3_target : float
        Absolute angles of the target equilibrium.
    p : 1-D array (13,)
        Packed parameter vector.

    Returns
    -------
    float
        Target physical energy E* (J).
    """
    gg1 = p[10]; gg2 = p[11]; gg3 = p[12]
    return -(gg1 * np.cos(phi1_target)
             + gg2 * np.cos(phi2_target)
             + gg3 * np.cos(phi3_target))
