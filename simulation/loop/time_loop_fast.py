"""Zero-allocation simulation loops using scalar-state RK4.

All loops operate on scalar state variables to eliminate array allocation
inside the hot path. Output arrays are pre-allocated once before the loop.
"""

import numpy as np
from numba import njit
from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast


@njit(cache=True)
def _angle_wrap(dx):
    """Wrap angle difference to [-pi, pi]."""
    while dx > np.pi:
        dx -= 2.0 * np.pi
    while dx < -np.pi:
        dx += 2.0 * np.pi
    return dx


@njit(cache=True, fastmath=True, boundscheck=False)
def _run_loop_fast(N, dt, q0, dq0, q_eq, K_flat, p, disturbance, u_max):
    """Zero-allocation simulation loop with scalar state.

    Returns (q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_saturated).
    u_raw_peak: maximum |u| before saturation clipping.
    n_saturated: number of steps where saturation was applied.
    """
    q_arr = np.empty((N, 4))
    dq_arr = np.empty((N, 4))
    u_ctrl_arr = np.empty(N)
    u_dist_arr = np.empty(N)
    u_raw_peak = 0.0
    n_saturated = 0

    # Unpack initial state to scalars
    sq0 = q0[0]; sq1 = q0[1]; sq2 = q0[2]; sq3 = q0[3]
    sdq0 = dq0[0]; sdq1 = dq0[1]; sdq2 = dq0[2]; sdq3 = dq0[3]
    eq0 = q_eq[0]; eq1 = q_eq[1]; eq2 = q_eq[2]; eq3 = q_eq[3]

    q_arr[0, 0] = sq0; q_arr[0, 1] = sq1; q_arr[0, 2] = sq2; q_arr[0, 3] = sq3
    dq_arr[0, 0] = sdq0; dq_arr[0, 1] = sdq1; dq_arr[0, 2] = sdq2; dq_arr[0, 3] = sdq3

    has_dist = disturbance.shape[0] > 0

    for k in range(N - 1):
        # Control: u = -K @ z with angle wrapping
        z0 = sq0 - eq0
        z1 = _angle_wrap(sq1 - eq1)
        z2 = _angle_wrap(sq2 - eq2)
        z3 = _angle_wrap(sq3 - eq3)
        u_c = -(K_flat[0] * z0 + K_flat[1] * z1 + K_flat[2] * z2 + K_flat[3] * z3
              + K_flat[4] * sdq0 + K_flat[5] * sdq1 + K_flat[6] * sdq2 + K_flat[7] * sdq3)

        # Track raw control before saturation
        u_abs = abs(u_c)
        if u_abs > u_raw_peak:
            u_raw_peak = u_abs

        # Saturation
        if u_c > u_max:
            u_c = u_max
            n_saturated += 1
        elif u_c < -u_max:
            u_c = -u_max
            n_saturated += 1
        u_ctrl_arr[k] = u_c

        u_d = 0.0
        if has_dist:
            u_d = disturbance[k]
        u_dist_arr[k] = u_d

        u_total = u_c + u_d

        # RK4 step (all scalar, zero allocation)
        sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u_total, p, dt)

        # NaN/divergence detection (check all state variables)
        if (sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3
                or sdq0 != sdq0 or sdq1 != sdq1 or sdq2 != sdq2 or sdq3 != sdq3):
            break

        q_arr[k + 1, 0] = sq0; q_arr[k + 1, 1] = sq1
        q_arr[k + 1, 2] = sq2; q_arr[k + 1, 3] = sq3
        dq_arr[k + 1, 0] = sdq0; dq_arr[k + 1, 1] = sdq1
        dq_arr[k + 1, 2] = sdq2; dq_arr[k + 1, 3] = sdq3

    # If simulation diverged (NaN break), fill ALL remainder with NaN
    if sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3:
        for fill_k in range(k + 1, N):
            q_arr[fill_k, 0] = np.nan; q_arr[fill_k, 1] = np.nan
            q_arr[fill_k, 2] = np.nan; q_arr[fill_k, 3] = np.nan
            dq_arr[fill_k, 0] = np.nan; dq_arr[fill_k, 1] = np.nan
            dq_arr[fill_k, 2] = np.nan; dq_arr[fill_k, 3] = np.nan
            u_ctrl_arr[fill_k] = np.nan
            u_dist_arr[fill_k] = np.nan
        return q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_saturated

    # Last step control
    z0 = sq0 - eq0
    z1 = _angle_wrap(sq1 - eq1)
    z2 = _angle_wrap(sq2 - eq2)
    z3 = _angle_wrap(sq3 - eq3)
    u_c = -(K_flat[0] * z0 + K_flat[1] * z1 + K_flat[2] * z2 + K_flat[3] * z3
          + K_flat[4] * sdq0 + K_flat[5] * sdq1 + K_flat[6] * sdq2 + K_flat[7] * sdq3)
    u_abs = abs(u_c)
    if u_abs > u_raw_peak:
        u_raw_peak = u_abs
    if u_c > u_max:
        u_c = u_max
        n_saturated += 1
    elif u_c < -u_max:
        u_c = -u_max
        n_saturated += 1
    u_ctrl_arr[N - 1] = u_c
    if has_dist and disturbance.shape[0] >= N:
        u_dist_arr[N - 1] = disturbance[N - 1]
    else:
        u_dist_arr[N - 1] = 0.0

    return q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_saturated


@njit(cache=True, fastmath=True, boundscheck=False)
def _run_loop_gs_fast(N, dt, q0, dq0, q_eq, p, disturbance,
                      gs_dev_angles, gs_K_gains, gs_slopes, u_max):
    """Zero-allocation gain-scheduled simulation loop with cubic Hermite interpolation.

    Returns (q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_saturated).
    """
    q_arr = np.empty((N, 4))
    dq_arr = np.empty((N, 4))
    u_ctrl_arr = np.empty(N)
    u_dist_arr = np.empty(N)
    u_raw_peak = 0.0
    n_saturated = 0

    sq0 = q0[0]; sq1 = q0[1]; sq2 = q0[2]; sq3 = q0[3]
    sdq0 = dq0[0]; sdq1 = dq0[1]; sdq2 = dq0[2]; sdq3 = dq0[3]
    eq0 = q_eq[0]; eq1 = q_eq[1]; eq2 = q_eq[2]; eq3 = q_eq[3]

    q_arr[0, 0] = sq0; q_arr[0, 1] = sq1; q_arr[0, 2] = sq2; q_arr[0, 3] = sq3
    dq_arr[0, 0] = sdq0; dq_arr[0, 1] = sdq1; dq_arr[0, 2] = sdq2; dq_arr[0, 3] = sdq3

    has_dist = disturbance.shape[0] > 0
    n_gs = gs_dev_angles.shape[0]

    for k in range(N - 1):
        # Gain interpolation
        delta = _angle_wrap(sq1 - eq1)
        # Find interval
        if delta <= gs_dev_angles[0]:
            K0 = gs_K_gains[0, 0]; K1 = gs_K_gains[0, 1]; K2 = gs_K_gains[0, 2]; K3 = gs_K_gains[0, 3]
            K4 = gs_K_gains[0, 4]; K5 = gs_K_gains[0, 5]; K6 = gs_K_gains[0, 6]; K7 = gs_K_gains[0, 7]
        elif delta >= gs_dev_angles[n_gs - 1]:
            K0 = gs_K_gains[n_gs-1, 0]; K1 = gs_K_gains[n_gs-1, 1]; K2 = gs_K_gains[n_gs-1, 2]; K3 = gs_K_gains[n_gs-1, 3]
            K4 = gs_K_gains[n_gs-1, 4]; K5 = gs_K_gains[n_gs-1, 5]; K6 = gs_K_gains[n_gs-1, 6]; K7 = gs_K_gains[n_gs-1, 7]
        else:
            idx = 0
            for i in range(n_gs - 1):
                if gs_dev_angles[i + 1] >= delta:
                    idx = i
                    break
            h_seg = gs_dev_angles[idx + 1] - gs_dev_angles[idx]
            t_interp = (delta - gs_dev_angles[idx]) / h_seg
            t2 = t_interp * t_interp
            t3 = t2 * t_interp
            # Hermite basis functions
            h00 = 2.0*t3 - 3.0*t2 + 1.0
            h10 = t3 - 2.0*t2 + t_interp
            h01 = -2.0*t3 + 3.0*t2
            h11 = t3 - t2
            K0 = h00*gs_K_gains[idx,0] + h10*h_seg*gs_slopes[idx,0] + h01*gs_K_gains[idx+1,0] + h11*h_seg*gs_slopes[idx+1,0]
            K1 = h00*gs_K_gains[idx,1] + h10*h_seg*gs_slopes[idx,1] + h01*gs_K_gains[idx+1,1] + h11*h_seg*gs_slopes[idx+1,1]
            K2 = h00*gs_K_gains[idx,2] + h10*h_seg*gs_slopes[idx,2] + h01*gs_K_gains[idx+1,2] + h11*h_seg*gs_slopes[idx+1,2]
            K3 = h00*gs_K_gains[idx,3] + h10*h_seg*gs_slopes[idx,3] + h01*gs_K_gains[idx+1,3] + h11*h_seg*gs_slopes[idx+1,3]
            K4 = h00*gs_K_gains[idx,4] + h10*h_seg*gs_slopes[idx,4] + h01*gs_K_gains[idx+1,4] + h11*h_seg*gs_slopes[idx+1,4]
            K5 = h00*gs_K_gains[idx,5] + h10*h_seg*gs_slopes[idx,5] + h01*gs_K_gains[idx+1,5] + h11*h_seg*gs_slopes[idx+1,5]
            K6 = h00*gs_K_gains[idx,6] + h10*h_seg*gs_slopes[idx,6] + h01*gs_K_gains[idx+1,6] + h11*h_seg*gs_slopes[idx+1,6]
            K7 = h00*gs_K_gains[idx,7] + h10*h_seg*gs_slopes[idx,7] + h01*gs_K_gains[idx+1,7] + h11*h_seg*gs_slopes[idx+1,7]

        z0 = sq0 - eq0
        z1 = _angle_wrap(sq1 - eq1)
        z2 = _angle_wrap(sq2 - eq2)
        z3 = _angle_wrap(sq3 - eq3)
        u_c = -(K0 * z0 + K1 * z1 + K2 * z2 + K3 * z3
              + K4 * sdq0 + K5 * sdq1 + K6 * sdq2 + K7 * sdq3)

        u_abs = abs(u_c)
        if u_abs > u_raw_peak:
            u_raw_peak = u_abs
        if u_c > u_max:
            u_c = u_max
            n_saturated += 1
        elif u_c < -u_max:
            u_c = -u_max
            n_saturated += 1
        u_ctrl_arr[k] = u_c

        u_d = 0.0
        if has_dist:
            u_d = disturbance[k]
        u_dist_arr[k] = u_d

        u_total = u_c + u_d
        sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u_total, p, dt)

        # NaN/divergence detection (all state variables)
        if (sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3
                or sdq0 != sdq0 or sdq1 != sdq1 or sdq2 != sdq2 or sdq3 != sdq3):
            break

        q_arr[k + 1, 0] = sq0; q_arr[k + 1, 1] = sq1
        q_arr[k + 1, 2] = sq2; q_arr[k + 1, 3] = sq3
        dq_arr[k + 1, 0] = sdq0; dq_arr[k + 1, 1] = sdq1
        dq_arr[k + 1, 2] = sdq2; dq_arr[k + 1, 3] = sdq3

    # If simulation diverged (NaN break), fill ALL remainder with NaN
    if sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3:
        for fill_k in range(k + 1, N):
            q_arr[fill_k, 0] = np.nan; q_arr[fill_k, 1] = np.nan
            q_arr[fill_k, 2] = np.nan; q_arr[fill_k, 3] = np.nan
            dq_arr[fill_k, 0] = np.nan; dq_arr[fill_k, 1] = np.nan
            dq_arr[fill_k, 2] = np.nan; dq_arr[fill_k, 3] = np.nan
            u_ctrl_arr[fill_k] = np.nan
            u_dist_arr[fill_k] = np.nan
        return q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_saturated

    # Last step
    delta = _angle_wrap(sq1 - eq1)
    if delta <= gs_dev_angles[0]:
        K0=gs_K_gains[0,0]; K1=gs_K_gains[0,1]; K2=gs_K_gains[0,2]; K3=gs_K_gains[0,3]
        K4=gs_K_gains[0,4]; K5=gs_K_gains[0,5]; K6=gs_K_gains[0,6]; K7=gs_K_gains[0,7]
    elif delta >= gs_dev_angles[n_gs-1]:
        K0=gs_K_gains[n_gs-1,0]; K1=gs_K_gains[n_gs-1,1]; K2=gs_K_gains[n_gs-1,2]; K3=gs_K_gains[n_gs-1,3]
        K4=gs_K_gains[n_gs-1,4]; K5=gs_K_gains[n_gs-1,5]; K6=gs_K_gains[n_gs-1,6]; K7=gs_K_gains[n_gs-1,7]
    else:
        idx = 0
        for i in range(n_gs-1):
            if gs_dev_angles[i+1] >= delta:
                idx = i
                break
        h_seg = gs_dev_angles[idx + 1] - gs_dev_angles[idx]
        t_interp = (delta - gs_dev_angles[idx]) / h_seg
        t2 = t_interp * t_interp
        t3 = t2 * t_interp
        h00 = 2.0*t3 - 3.0*t2 + 1.0
        h10 = t3 - 2.0*t2 + t_interp
        h01 = -2.0*t3 + 3.0*t2
        h11 = t3 - t2
        K0 = h00*gs_K_gains[idx,0] + h10*h_seg*gs_slopes[idx,0] + h01*gs_K_gains[idx+1,0] + h11*h_seg*gs_slopes[idx+1,0]
        K1 = h00*gs_K_gains[idx,1] + h10*h_seg*gs_slopes[idx,1] + h01*gs_K_gains[idx+1,1] + h11*h_seg*gs_slopes[idx+1,1]
        K2 = h00*gs_K_gains[idx,2] + h10*h_seg*gs_slopes[idx,2] + h01*gs_K_gains[idx+1,2] + h11*h_seg*gs_slopes[idx+1,2]
        K3 = h00*gs_K_gains[idx,3] + h10*h_seg*gs_slopes[idx,3] + h01*gs_K_gains[idx+1,3] + h11*h_seg*gs_slopes[idx+1,3]
        K4 = h00*gs_K_gains[idx,4] + h10*h_seg*gs_slopes[idx,4] + h01*gs_K_gains[idx+1,4] + h11*h_seg*gs_slopes[idx+1,4]
        K5 = h00*gs_K_gains[idx,5] + h10*h_seg*gs_slopes[idx,5] + h01*gs_K_gains[idx+1,5] + h11*h_seg*gs_slopes[idx+1,5]
        K6 = h00*gs_K_gains[idx,6] + h10*h_seg*gs_slopes[idx,6] + h01*gs_K_gains[idx+1,6] + h11*h_seg*gs_slopes[idx+1,6]
        K7 = h00*gs_K_gains[idx,7] + h10*h_seg*gs_slopes[idx,7] + h01*gs_K_gains[idx+1,7] + h11*h_seg*gs_slopes[idx+1,7]

    z0 = sq0-eq0; z1 = _angle_wrap(sq1-eq1); z2 = _angle_wrap(sq2-eq2); z3 = _angle_wrap(sq3-eq3)
    u_c = -(K0*z0+K1*z1+K2*z2+K3*z3+K4*sdq0+K5*sdq1+K6*sdq2+K7*sdq3)
    u_abs = abs(u_c)
    if u_abs > u_raw_peak:
        u_raw_peak = u_abs
    if u_c > u_max:
        u_c = u_max
        n_saturated += 1
    elif u_c < -u_max:
        u_c = -u_max
        n_saturated += 1
    u_ctrl_arr[N-1] = u_c
    if has_dist and disturbance.shape[0] >= N:
        u_dist_arr[N-1] = disturbance[N-1]
    else:
        u_dist_arr[N-1] = 0.0

    return q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_saturated
