"""Compute Cartesian positions of the pendulum joints."""

import numpy as np


def compute_joint_positions(cart_x, phi1, phi2, phi3, L1, L2, L3) -> dict:
    """Return dict with P1x, P1y, P2x, P2y, P3x, P3y arrays."""
    P1x = cart_x + L1 * np.sin(phi1)
    P1y = -L1 * np.cos(phi1)

    P2x = P1x + L2 * np.sin(phi2)
    P2y = P1y - L2 * np.cos(phi2)

    P3x = P2x + L3 * np.sin(phi3)
    P3y = P2y - L3 * np.cos(phi3)

    return dict(P1x=P1x, P1y=P1y, P2x=P2x, P2y=P2y, P3x=P3x, P3y=P3y)
