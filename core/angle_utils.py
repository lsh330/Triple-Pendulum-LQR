"""Shared angle-wrapping utility for the triple pendulum simulation.

Consolidates the duplicate _angle_wrap implementations previously found in
simulation/loop/time_loop_fast.py and analysis/region_of_attraction.py.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def angle_wrap(dx):
    """Wrap angle difference to [-pi, pi]. O(1) via floor division."""
    dx = dx - 2.0 * np.pi * np.floor((dx + np.pi) / (2.0 * np.pi))
    return dx
