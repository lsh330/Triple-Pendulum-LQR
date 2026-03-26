"""Compute absolute angles from generalised coordinates."""

import numpy as np


def compute_absolute_angles(q: np.ndarray):
    """Return (phi1, phi2, phi3) -- absolute angles of each link.

    Parameters
    ----------
    q : ndarray, shape (N, 4)
        Columns: cart_x, theta1, theta2, theta3
    """
    phi1 = q[:, 1]
    phi2 = q[:, 1] + q[:, 2]
    phi3 = q[:, 1] + q[:, 2] + q[:, 3]
    return phi1, phi2, phi3
