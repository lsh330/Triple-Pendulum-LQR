"""Gain scheduling for the cart + triple pendulum.

Precomputes LQR gains at multiple operating points (theta1 deviations from
upright) and linearly interpolates at runtime based on the current state.

The interpolation data is packed as plain numpy arrays so that a @njit
function can perform the lookup inside the JIT-compiled simulation loop.
"""

import numpy as np
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati
from control.gain_computation.compute_K import compute_K


class GainScheduler:
    """Precomputes LQR gains at several operating points and interpolates.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration.
    deviation_angles_deg : list of float
        Theta1 deviations (degrees) from the upright equilibrium at which
        to linearize and solve LQR.
    """

    def __init__(self, cfg, deviation_angles_deg=None):
        if deviation_angles_deg is None:
            deviation_angles_deg = [-20, -10, -5, 0, 5, 10, 20]

        self.cfg = cfg
        p = cfg.pack()
        q_eq = cfg.equilibrium

        Q = default_Q()
        R = default_R()

        dev_rad = np.deg2rad(np.array(deviation_angles_deg, dtype=np.float64))
        n_pts = len(dev_rad)

        # Sort deviations for correct interpolation
        sort_idx = np.argsort(dev_rad)
        dev_rad = dev_rad[sort_idx]

        K_gains = np.zeros((n_pts, 8))

        for i, delta in enumerate(dev_rad):
            q_op = q_eq.copy()
            q_op[1] += delta  # perturb theta1

            A, B = linearize(q_op, p)
            P = solve_riccati(A, B, Q, R)
            Ki = compute_K(R, B, P)
            K_gains[i] = Ki.flatten()

        # Store as arrays for JIT-compatible interpolation
        self.deviation_angles = dev_rad        # (n_pts,) sorted
        self.K_gains = K_gains                 # (n_pts, 8)

    def get_gain(self, q, q_eq):
        """Return an interpolated 1-D gain vector K (8,) for the current state.

        Parameters
        ----------
        q : ndarray (4,)
            Current joint configuration.
        q_eq : ndarray (4,)
            Equilibrium configuration.

        Returns
        -------
        K : ndarray (8,)
            Interpolated LQR gain vector.
        """
        return _interp_gain(q, q_eq, self.deviation_angles, self.K_gains)

    def pack_for_njit(self):
        """Return (deviation_angles, K_gains) as plain arrays for @njit."""
        return self.deviation_angles.copy(), self.K_gains.copy()


def _interp_gain(q, q_eq, dev_angles, K_gains):
    """Linear interpolation of gain based on theta1 deviation.

    Pure numpy -- no Python objects, compatible with being called from
    a thin wrapper around @njit code.
    """
    delta = q[1] - q_eq[1]  # theta1 deviation from equilibrium

    # Clamp to the range of precomputed angles
    if delta <= dev_angles[0]:
        return K_gains[0].copy()
    if delta >= dev_angles[-1]:
        return K_gains[-1].copy()

    # Find bracketing interval
    idx = np.searchsorted(dev_angles, delta) - 1
    idx = max(0, min(idx, len(dev_angles) - 2))

    alpha_lo = dev_angles[idx]
    alpha_hi = dev_angles[idx + 1]
    t = (delta - alpha_lo) / (alpha_hi - alpha_lo)

    return (1.0 - t) * K_gains[idx] + t * K_gains[idx + 1]
