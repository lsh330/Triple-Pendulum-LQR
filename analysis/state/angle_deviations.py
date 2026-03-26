"""Compute angle deviations from equilibrium."""

import numpy as np


def compute_angle_deviations(q: np.ndarray, q_eq: np.ndarray):
    """Return (dth1, dth2, dth3) deviations from equilibrium angles.

    Parameters
    ----------
    q : ndarray, shape (N, 4)
    q_eq : ndarray, shape (4,)
    """
    dth1 = q[:, 1] - q_eq[1]
    dth2 = q[:, 2] - q_eq[2]
    dth3 = q[:, 3] - q_eq[3]
    return dth1, dth2, dth3
