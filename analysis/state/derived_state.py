"""Combine all state analysis into a single derived-state dictionary."""

import numpy as np

from analysis.state.absolute_angles import compute_absolute_angles
from analysis.state.joint_positions import compute_joint_positions
from analysis.state.angle_deviations import compute_angle_deviations


def compute_derived_state(cfg, q: np.ndarray, dq: np.ndarray) -> dict:
    """Return a dict containing cart_x, absolute angles, joint positions,
    and angle deviations."""
    cart_x = q[:, 0]
    phi1, phi2, phi3 = compute_absolute_angles(q)

    positions = compute_joint_positions(
        cart_x, phi1, phi2, phi3, cfg.L1, cfg.L2, cfg.L3
    )

    dth1, dth2, dth3 = compute_angle_deviations(q, cfg.equilibrium)

    return dict(
        cart_x=cart_x,
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
        dth1=dth1,
        dth2=dth2,
        dth3=dth3,
        **positions,
    )
