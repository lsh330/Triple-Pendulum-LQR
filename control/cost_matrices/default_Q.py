import numpy as np


def default_Q():
    """Return the default 8x8 state cost matrix."""
    return np.diag([10.0, 100.0, 100.0, 100.0, 1.0, 10.0, 10.0, 10.0])


def adaptive_Q(cfg):
    """Return a Q matrix scaled by system inertia for parameter-adaptive tuning.

    Uses Bryson's rule: Q_ii = 1 / (max_acceptable_deviation_i)^2,
    with deviations scaled by link inertias.
    """
    # Cart position: scale by total mass (heavier = slower response)
    Mt = cfg.mc + cfg.m1 + cfg.m2 + cfg.m3
    # Bryson's rule: scale cart penalty by total mass relative to baseline
    # For Medrano-Cerda (Mt≈6kg): q_cart≈10. For heavier systems: proportionally more
    q_cart = max(1.0, 10.0 * np.sqrt(Mt / max(cfg.mc, 0.1)))

    # Angles: scale by inverse of link inertia (lighter links need more penalty)
    I1 = cfg.m1 * cfg.L1**2 / 3.0
    I2 = cfg.m2 * cfg.L2**2 / 3.0
    I3 = cfg.m3 * cfg.L3**2 / 3.0
    I_ref = (I1 + I2 + I3) / 3.0
    q_th1 = 100.0 * I_ref / max(I1, 1e-6)
    q_th2 = 100.0 * I_ref / max(I2, 1e-6)
    q_th3 = 100.0 * I_ref / max(I3, 1e-6)

    # Velocities: moderate damping scaled similarly
    q_dcart = 1.0
    q_dth1 = 10.0 * I_ref / max(I1, 1e-6)
    q_dth2 = 10.0 * I_ref / max(I2, 1e-6)
    q_dth3 = 10.0 * I_ref / max(I3, 1e-6)

    return np.diag([q_cart, q_th1, q_th2, q_th3,
                    q_dcart, q_dth1, q_dth2, q_dth3])
