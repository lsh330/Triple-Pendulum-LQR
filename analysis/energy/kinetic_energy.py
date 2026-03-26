"""Compute total kinetic energy of the cart + triple pendulum."""

import numpy as np


def compute_kinetic_energy(cfg, q, dq, phi1, phi2, phi3) -> np.ndarray:
    """Return KE array (N,).

    Includes translational KE of the cart, translational + rotational KE of
    each link (point-mass at CoM = half-length, thin-rod inertia = mL^2/12).
    """
    dx = dq[:, 0]
    dphi1 = dq[:, 1]
    dphi2 = dq[:, 1] + dq[:, 2]
    dphi3 = dq[:, 1] + dq[:, 2] + dq[:, 3]

    # Cart
    KE = 0.5 * cfg.mc * dx ** 2

    # --- Link 1 ---
    lc1 = cfg.L1 / 2.0
    vx1 = dx + lc1 * np.cos(phi1) * dphi1
    vy1 = -lc1 * np.sin(phi1) * dphi1
    I1 = cfg.m1 * cfg.L1 ** 2 / 12.0
    KE += 0.5 * cfg.m1 * (vx1 ** 2 + vy1 ** 2) + 0.5 * I1 * dphi1 ** 2

    # --- Link 2 ---
    lc2 = cfg.L2 / 2.0
    # Velocity of joint 1
    jx1 = dx + cfg.L1 * np.cos(phi1) * dphi1
    jy1 = -cfg.L1 * np.sin(phi1) * dphi1
    vx2 = jx1 + lc2 * np.cos(phi2) * dphi2
    vy2 = jy1 - lc2 * np.sin(phi2) * dphi2
    I2 = cfg.m2 * cfg.L2 ** 2 / 12.0
    KE += 0.5 * cfg.m2 * (vx2 ** 2 + vy2 ** 2) + 0.5 * I2 * dphi2 ** 2

    # --- Link 3 ---
    lc3 = cfg.L3 / 2.0
    jx2 = jx1 + cfg.L2 * np.cos(phi2) * dphi2
    jy2 = jy1 - cfg.L2 * np.sin(phi2) * dphi2
    vx3 = jx2 + lc3 * np.cos(phi3) * dphi3
    vy3 = jy2 - lc3 * np.sin(phi3) * dphi3
    I3 = cfg.m3 * cfg.L3 ** 2 / 12.0
    KE += 0.5 * cfg.m3 * (vx3 ** 2 + vy3 ** 2) + 0.5 * I3 * dphi3 ** 2

    return KE
