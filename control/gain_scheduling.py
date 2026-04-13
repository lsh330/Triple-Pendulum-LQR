"""Gain scheduling with multi-axis support and smooth interpolation.

Provides two schedulers:
- GainScheduler: 1D interpolation on theta1 with cubic Hermite spline
- MultiAxisGainScheduler: 3D trilinear interpolation on (theta1, theta2, theta3)

Both pack their data for @njit-compatible runtime lookup.

W4 수정사항
-----------
``_compute_gain_at`` 이 trim solver 를 사용해 격자점 u_eq 피드포워드를
계산한다. 기존 heuristic offset(u=0) 대비 격자점에서 중력 불균형이
정확히 보상된다.  GainScheduler / MultiAxisGainScheduler 는 u_ff 배열을
함께 저장하여 ``get_gain_and_ff`` 로 반환한다.
"""

import numpy as np
from numba import njit
from control.linearization.linearize import linearize
from control.linearization.trim_solver import compute_trim_u
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati
from control.gain_computation.compute_K import compute_K


def _compute_gain_at(cfg, q_offset, Q=None, R=None):
    """격자점 equilibrium + offset 에서 LQR 게인과 트림 피드포워드를 계산한다.

    W4: trim solver 를 호출해 G(q) = B_u * u_eq 를 만족하는 u_eq 를 계산.
    기존에는 u=0 heuristic 이었으나 이제 중력 불균형이 명시적으로 보상된다.

    Returns
    -------
    K_flat : (8,) float array
    u_ff   : float  — trim feedforward (N)
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
    K_flat = compute_K(R, B, P).flatten()
    # W4: 트림 피드포워드 계산
    u_ff = compute_trim_u(q_op, p)
    return K_flat, u_ff


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
        # W4: 트림 피드포워드 벡터 저장
        u_ff_gains = np.zeros(n_pts)
        for i, delta in enumerate(dev_rad):
            K_flat, u_ff = _compute_gain_at(cfg, np.array([delta, 0.0, 0.0]), Q, R)
            K_gains[i] = K_flat
            u_ff_gains[i] = u_ff

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

        # W4: u_ff 1D cubic Hermite slopes
        slopes_ff = np.zeros(n_pts)
        for i in range(n_pts):
            if i == 0:
                slopes_ff[i] = (u_ff_gains[1] - u_ff_gains[0]) / (dev_rad[1] - dev_rad[0])
            elif i == n_pts - 1:
                slopes_ff[i] = (u_ff_gains[-1] - u_ff_gains[-2]) / (dev_rad[-1] - dev_rad[-2])
            else:
                h_l = dev_rad[i] - dev_rad[i-1]
                h_r = dev_rad[i+1] - dev_rad[i]
                d_l = (u_ff_gains[i] - u_ff_gains[i-1]) / h_l
                d_r = (u_ff_gains[i+1] - u_ff_gains[i]) / h_r
                if d_l * d_r > 0:
                    w1 = 2.0*h_r + h_l
                    w2 = h_r + 2.0*h_l
                    slopes_ff[i] = (w1 + w2) / (w1/d_l + w2/d_r)
                else:
                    slopes_ff[i] = 0.0

        self.deviation_angles = dev_rad
        self.K_gains = K_gains
        self.slopes = slopes
        # W4: 트림 피드포워드 데이터
        self.u_ff_gains = u_ff_gains
        self.slopes_ff = slopes_ff

    def get_gain(self, q, q_eq):
        """Return interpolated gain vector K (8,) for current state."""
        delta = q[1] - q_eq[1]
        return _interp_cubic(delta, self.deviation_angles, self.K_gains, self.slopes)

    def get_gain_and_ff(self, q, q_eq):
        """W4: 보간된 게인 K(8,) 와 트림 피드포워드 u_ff(float) 를 반환한다."""
        delta = q[1] - q_eq[1]
        K = _interp_cubic(delta, self.deviation_angles, self.K_gains, self.slopes)
        u_ff = _interp_scalar(delta, self.deviation_angles,
                              self.u_ff_gains, self.slopes_ff)
        return K, u_ff

    def pack_for_njit(self):
        """Return (dev_angles, K_gains, slopes) for @njit cubic Hermite interpolation."""
        return self.deviation_angles.copy(), self.K_gains.copy(), self.slopes.copy()

    def pack_ff_for_njit(self):
        """W4: (dev_angles, u_ff_gains, slopes_ff) 를 반환한다."""
        return self.deviation_angles.copy(), self.u_ff_gains.copy(), self.slopes_ff.copy()


@njit(cache=True)
def _interp_scalar(delta, dev_angles, values, slopes):
    """W4: 스칼라 값(예: u_ff)에 대한 1D cubic Hermite 보간."""
    n = dev_angles.shape[0]
    if delta <= dev_angles[0]:
        return values[0]
    if delta >= dev_angles[n - 1]:
        return values[n - 1]

    idx = 0
    for i in range(n - 1):
        if dev_angles[i + 1] >= delta:
            idx = i
            break

    h = dev_angles[idx + 1] - dev_angles[idx]
    t = (delta - dev_angles[idx]) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0*t3 - 3.0*t2 + 1.0
    h10 = t3 - 2.0*t2 + t
    h01 = -2.0*t3 + 3.0*t2
    h11 = t3 - t2

    return (h00 * values[idx] + h10 * h * slopes[idx]
            + h01 * values[idx + 1] + h11 * h * slopes[idx + 1])


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
    gains at each vertex. The Python-level get_gain() uses full trilinear
    interpolation. The JIT fast loop uses a 1D cubic Hermite projection
    (theta2=theta3=0 center slice) for zero-overhead scalar operation.
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
        # W4: 3D trim 피드포워드 그리드
        u_ff_grid = np.zeros((n1, n2, n3))

        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    offset = np.array([t1[i], t2[j], t3[k]])
                    K_flat, u_ff = _compute_gain_at(cfg, offset, Q, R)
                    K_grid[i, j, k] = K_flat
                    u_ff_grid[i, j, k] = u_ff

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.K_grid = K_grid
        # W4: trim 피드포워드 그리드
        self.u_ff_grid = u_ff_grid

    def get_gain(self, q, q_eq):
        """Return trilinear interpolated gain vector K (8,)."""
        d1 = q[1] - q_eq[1]
        d2 = q[2] - q_eq[2]
        d3 = q[3] - q_eq[3]
        return _interp_trilinear(d1, d2, d3, self.t1, self.t2, self.t3, self.K_grid)

    def get_gain_and_ff(self, q, q_eq):
        """W4: 트라이리니어 보간 게인 K(8,) 와 트림 u_ff(float) 반환."""
        d1 = q[1] - q_eq[1]
        d2 = q[2] - q_eq[2]
        d3 = q[3] - q_eq[3]
        K = _interp_trilinear(d1, d2, d3, self.t1, self.t2, self.t3, self.K_grid)
        u_ff = float(_interp_trilinear_scalar(
            d1, d2, d3, self.t1, self.t2, self.t3, self.u_ff_grid))
        return K, u_ff

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
def _interp_trilinear_scalar(d1, d2, d3, t1, t2, t3, val_grid):
    """W4: 3D 스칼라 그리드(예: u_ff_grid)에 대한 trilinear 보간."""
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

    c000 = val_grid[i1,   i2,   i3]
    c100 = val_grid[i1+1, i2,   i3]
    c010 = val_grid[i1,   i2+1, i3]
    c110 = val_grid[i1+1, i2+1, i3]
    c001 = val_grid[i1,   i2,   i3+1]
    c101 = val_grid[i1+1, i2,   i3+1]
    c011 = val_grid[i1,   i2+1, i3+1]
    c111 = val_grid[i1+1, i2+1, i3+1]

    c00 = c000*(1-a1) + c100*a1
    c01 = c001*(1-a1) + c101*a1
    c10 = c010*(1-a1) + c110*a1
    c11 = c011*(1-a1) + c111*a1

    c0 = c00*(1-a2) + c10*a2
    c1 = c01*(1-a2) + c11*a2

    return c0*(1-a3) + c1*a3


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
