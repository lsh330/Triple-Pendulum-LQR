"""Compute gravitational potential energy of the triple pendulum."""

import numpy as np


def compute_potential_energy(cfg, phi1, phi2, phi3) -> np.ndarray:
    """Return PE array (N,).

    Reference is the cart rail (y = 0).  CoM of each link is at its
    half-length above the lower joint.
    """
    g = cfg.g

    # CoM heights (measured from cart pivot)
    h1 = (cfg.L1 / 2.0) * np.cos(phi1)
    h2 = cfg.L1 * np.cos(phi1) + (cfg.L2 / 2.0) * np.cos(phi2)
    h3 = (cfg.L1 * np.cos(phi1) + cfg.L2 * np.cos(phi2)
           + (cfg.L3 / 2.0) * np.cos(phi3))

    PE = cfg.m1 * g * h1 + cfg.m2 * g * h2 + cfg.m3 * g * h3
    return PE
