"""Gain scheduling with multi-axis support and smooth interpolation.

Provides two schedulers:
- GainScheduler: 1D interpolation on theta1 with cubic Hermite spline
- MultiAxisGainScheduler: 3D trilinear interpolation on (theta1, theta2, theta3)

Both pack their data for @njit-compatible runtime lookup.
"""

import numpy as np
from numba import njit
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati
from control.gain_computation.compute_K import compute_K


def _compute_gain_at(cfg, q_offset, Q=None, R=None):
    """Compute LQR gain at equilibrium + offset.

    NOTE: The offset operating points are NOT true equilibria — at these
    configurations with dq=0 and u=0, gravity produces nonzero torques.
    This is a heuristic local gain bank that works well for small deviations
    from the true upright equilibrium. The linearization accounts for the
    gravity gradient (dG/dq) at the offset point, which provides reasonable
    local stabilization despite the theoretical limitation.
    """
    p = cfg.pack()
    q_op = cfg.equilibrium.copy()
    q_op[1:4] += q_offset[:3]
    if Q is None:
        Q = default_Q()
    if R is None:
        R = default_R()
    A, B = linearize(q_op, p)
    P = solve_riccati(A, B, Q, R)
    return compute_K(R, B, P).flatten()


class GainScheduler:
    """1D gain scheduling on theta1 with cubic Hermite interpolation.

    Precomputes LQR gains at multiple theta1 deviations and interpolates
    at runtime using monotone piecewise cubic Hermite (PCHIP-like) spline.

    Warning: Operating points are heuristic perturbations of the upright
    equilibrium, NOT true equilibria. The gravity torque is nonzero at
    these points. For rigorous gain scheduling, a trim solver that finds
    u_eq(q) satisfying M*0 + G(q) = B*u_eq would be needed.
    """

    def __init__(self, cfg, deviation_angles_deg=None):
        if deviation_angles_deg is None:
            deviation_angles_deg = [-20, -10, -5, 0, 5, 10, 20]

        self.cfg = cfg
        Q = default_Q()
        R = default_R()

        dev_rad = np.deg2rad(np.array(deviation_angles_deg, dtype=np.float64))
        sort_idx = np.argsort(dev_rad)
        dev_rad = dev_rad[sort_idx]
        n_pts = len(dev_rad)

        K_gains = np.zeros((n_pts, 8))
        for i, delta in enumerate(dev_rad):
            K_gains[i] = _compute_gain_at(cfg, np.array([delta, 0.0, 0.0]), Q, R)

        # Precompute cubic Hermite slopes using finite differences
        # Uses 3-point formula at interior, one-sided at boundaries
        slopes = np.zeros((n_pts, 8))
        for j in range(8):
            for i in range(n_pts):
                if i == 0:
                    slopes[i, j] = (K_gains[1, j] - K_gains[0, j]) / (dev_rad[1] - dev_rad[0])
                elif i == n_pts - 1:
                    slopes[i, j] = (K_gains[-1, j] - K_gains[-2, j]) / (dev_rad[-1] - dev_rad[-2])
                else:
                    h_left = dev_rad[i] - dev_rad[i-1]
                    h_right = dev_rad[i+1] - dev_rad[i]
                    d_left = (K_gains[i, j] - K_gains[i-1, j]) / h_left
                    d_right = (K_gains[i+1, j] - K_gains[i, j]) / h_right
                    # Weighted harmonic mean (Fritsch-Carlson)
                    if d_left * d_right > 0:
                        w1 = 2.0 * h_right + h_left
                        w2 = h_right + 2.0 * h_left
                        slopes[i, j] = (w1 + w2) / (w1/d_left + w2/d_right)
                    else:
                        slopes[i, j] = 0.0

        self.deviation_angles = dev_rad
        self.K_gains = K_gains
        self.slopes = slopes

    def get_gain(self, q, q_eq):
        """Return interpolated gain vector K (8,) for current state."""
        delta = q[1] - q_eq[1]
        return _interp_cubic(delta, self.deviation_angles, self.K_gains, self.slopes)

    def pack_for_njit(self):
        """Return (dev_angles, K_gains, slopes) for @njit cubic Hermite interpolation."""
        return self.deviation_angles.copy(), self.K_gains.copy(), self.slopes.copy()


@njit(cache=True)
def _interp_cubic(delta, dev_angles, K_gains, slopes):
    """Cubic Hermite interpolation of gain vector."""
    n = dev_angles.shape[0]
    if delta <= dev_angles[0]:
        return K_gains[0].copy()
    if delta >= dev_angles[n - 1]:
        return K_gains[n - 1].copy()

    idx = 0
    for i in range(n - 1):
        if dev_angles[i + 1] >= delta:
            idx = i
            break

    h = dev_angles[idx + 1] - dev_angles[idx]
    t = (delta - dev_angles[idx]) / h
    t2 = t * t
    t3 = t2 * t

    # Hermite basis functions
    h00 = 2.0*t3 - 3.0*t2 + 1.0
    h10 = t3 - 2.0*t2 + t
    h01 = -2.0*t3 + 3.0*t2
    h11 = t3 - t2

    K_out = np.empty(8)
    for j in range(8):
        K_out[j] = (h00 * K_gains[idx, j] + h10 * h * slopes[idx, j]
                   + h01 * K_gains[idx+1, j] + h11 * h * slopes[idx+1, j])
    return K_out


class MultiAxisGainScheduler:
    """3D gain scheduling on (theta1, theta2, theta3) with trilinear interpolation.

    Creates a tensor-product grid of operating points and precomputes LQR
    gains at each vertex. Runtime lookup uses trilinear interpolation.
    """

    def __init__(self, cfg,
                 theta1_deg=None, theta2_deg=None, theta3_deg=None):
        if theta1_deg is None:
            theta1_deg = [-20, -12, -5, 0, 5, 12, 20]
        if theta2_deg is None:
            theta2_deg = [-10, -4, 0, 4, 10]
        if theta3_deg is None:
            theta3_deg = [-6, -2, 0, 2, 6]

        self.cfg = cfg
        Q = default_Q()
        R = default_R()

        t1 = np.deg2rad(np.sort(np.array(theta1_deg, dtype=np.float64)))
        t2 = np.deg2rad(np.sort(np.array(theta2_deg, dtype=np.float64)))
        t3 = np.deg2rad(np.sort(np.array(theta3_deg, dtype=np.float64)))

        n1, n2, n3 = len(t1), len(t2), len(t3)
        K_grid = np.zeros((n1, n2, n3, 8))

        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    offset = np.array([t1[i], t2[j], t3[k]])
                    K_grid[i, j, k] = _compute_gain_at(cfg, offset, Q, R)

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.K_grid = K_grid

    def get_gain(self, q, q_eq):
        """Return trilinear interpolated gain vector K (8,)."""
        d1 = q[1] - q_eq[1]
        d2 = q[2] - q_eq[2]
        d3 = q[3] - q_eq[3]
        return _interp_trilinear(d1, d2, d3, self.t1, self.t2, self.t3, self.K_grid)

    def pack_for_njit(self):
        """Pack as 1D projection with cubic Hermite slopes for fast loops."""
        n2_mid = len(self.t2) // 2
        n3_mid = len(self.t3) // 2
        dev = self.t1.copy()
        K_gains = self.K_grid[:, n2_mid, n3_mid, :].copy()
        # Compute Fritsch-Carlson monotone slopes for projected 1D gains
        n_pts = len(dev)
        slopes = np.zeros((n_pts, 8))
        for j in range(8):
            for i in range(n_pts):
                if i == 0:
                    slopes[i, j] = (K_gains[1, j] - K_gains[0, j]) / (dev[1] - dev[0])
                elif i == n_pts - 1:
                    slopes[i, j] = (K_gains[-1, j] - K_gains[-2, j]) / (dev[-1] - dev[-2])
                else:
                    h_l = dev[i] - dev[i-1]
                    h_r = dev[i+1] - dev[i]
                    d_l = (K_gains[i, j] - K_gains[i-1, j]) / h_l
                    d_r = (K_gains[i+1, j] - K_gains[i, j]) / h_r
                    if d_l * d_r > 0:
                        w1 = 2.0*h_r + h_l
                        w2 = h_r + 2.0*h_l
                        slopes[i, j] = (w1 + w2) / (w1/d_l + w2/d_r)
                    else:
                        slopes[i, j] = 0.0
        return dev, K_gains, slopes


@njit(cache=True)
def _interp_trilinear(d1, d2, d3, t1, t2, t3, K_grid):
    """Trilinear interpolation on 3D gain grid."""
    # Clamp and find indices for each axis
    def _find_idx(val, arr):
        n = arr.shape[0]
        if val <= arr[0]:
            return 0, 0.0
        if val >= arr[n - 1]:
            return n - 2, 1.0
        idx = 0
        for i in range(n - 1):
            if arr[i + 1] >= val:
                idx = i
                break
        t = (val - arr[idx]) / (arr[idx + 1] - arr[idx])
        return idx, t

    i1, a1 = _find_idx(d1, t1)
    i2, a2 = _find_idx(d2, t2)
    i3, a3 = _find_idx(d3, t3)

    K_out = np.empty(8)
    for j in range(8):
        # Trilinear: 8 vertices
        c000 = K_grid[i1, i2, i3, j]
        c100 = K_grid[i1+1, i2, i3, j]
        c010 = K_grid[i1, i2+1, i3, j]
        c110 = K_grid[i1+1, i2+1, i3, j]
        c001 = K_grid[i1, i2, i3+1, j]
        c101 = K_grid[i1+1, i2, i3+1, j]
        c011 = K_grid[i1, i2+1, i3+1, j]
        c111 = K_grid[i1+1, i2+1, i3+1, j]

        c00 = c000*(1-a1) + c100*a1
        c01 = c001*(1-a1) + c101*a1
        c10 = c010*(1-a1) + c110*a1
        c11 = c011*(1-a1) + c111*a1

        c0 = c00*(1-a2) + c10*a2
        c1 = c01*(1-a2) + c11*a2

        K_out[j] = c0*(1-a3) + c1*a3

    return K_out
