"""Main simulation loop -- fully JIT-compiled for maximum performance."""

import numpy as np
from numba import njit
from simulation.integrator.rk4_step import rk4_step
from simulation.loop.control_law import compute_control
from simulation.initial_conditions.impulse_response import apply_impulse
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def _run_loop(N, dt, q0, dq0, q_eq, K_flat, p, disturbance):
    """JIT-compiled simulation loop. No Python overhead per step."""
    q_arr = np.empty((N, 4))
    dq_arr = np.empty((N, 4))
    u_ctrl_arr = np.empty(N)
    u_dist_arr = np.empty(N)

    q = q0.copy()
    dq = dq0.copy()
    q_arr[0] = q
    dq_arr[0] = dq

    has_dist = disturbance.shape[0] > 0

    for k in range(N - 1):
        u_c = compute_control(q, dq, q_eq, K_flat)
        u_ctrl_arr[k] = u_c

        u_d = 0.0
        if has_dist:
            u_d = disturbance[k]
        u_dist_arr[k] = u_d

        u_total = u_c + u_d
        q, dq = rk4_step(q, dq, u_total, p, dt)
        q_arr[k + 1] = q
        dq_arr[k + 1] = dq

    # Last step control
    u_ctrl_arr[N - 1] = compute_control(q, dq, q_eq, K_flat)
    if has_dist and disturbance.shape[0] >= N:
        u_dist_arr[N - 1] = disturbance[N - 1]
    else:
        u_dist_arr[N - 1] = 0.0

    return q_arr, dq_arr, u_ctrl_arr, u_dist_arr


@njit(cache=True)
def _interp_gain_njit(delta, dev_angles, K_gains):
    """Linearly interpolate gain inside @njit based on theta1 deviation."""
    n = dev_angles.shape[0]
    if delta <= dev_angles[0]:
        return K_gains[0].copy()
    if delta >= dev_angles[n - 1]:
        return K_gains[n - 1].copy()

    # Binary-style search (simple linear scan is fine for ~7 points)
    idx = 0
    for i in range(n - 1):
        if dev_angles[i + 1] >= delta:
            idx = i
            break

    alpha_lo = dev_angles[idx]
    alpha_hi = dev_angles[idx + 1]
    t = (delta - alpha_lo) / (alpha_hi - alpha_lo)

    K_out = np.empty(8)
    for j in range(8):
        K_out[j] = (1.0 - t) * K_gains[idx, j] + t * K_gains[idx + 1, j]
    return K_out


@njit(cache=True)
def _run_loop_gs(N, dt, q0, dq0, q_eq, p, disturbance,
                 gs_dev_angles, gs_K_gains):
    """JIT-compiled simulation loop with gain-scheduled control."""
    q_arr = np.empty((N, 4))
    dq_arr = np.empty((N, 4))
    u_ctrl_arr = np.empty(N)
    u_dist_arr = np.empty(N)

    q = q0.copy()
    dq = dq0.copy()
    q_arr[0] = q
    dq_arr[0] = dq

    has_dist = disturbance.shape[0] > 0

    for k in range(N - 1):
        delta = q[1] - q_eq[1]
        K_flat = _interp_gain_njit(delta, gs_dev_angles, gs_K_gains)
        u_c = compute_control(q, dq, q_eq, K_flat)
        u_ctrl_arr[k] = u_c

        u_d = 0.0
        if has_dist:
            u_d = disturbance[k]
        u_dist_arr[k] = u_d

        u_total = u_c + u_d
        q, dq = rk4_step(q, dq, u_total, p, dt)
        q_arr[k + 1] = q
        dq_arr[k + 1] = dq

    # Last step control
    delta = q[1] - q_eq[1]
    K_flat = _interp_gain_njit(delta, gs_dev_angles, gs_K_gains)
    u_ctrl_arr[N - 1] = compute_control(q, dq, q_eq, K_flat)
    if has_dist and disturbance.shape[0] >= N:
        u_dist_arr[N - 1] = disturbance[N - 1]
    else:
        u_dist_arr[N - 1] = 0.0

    return q_arr, dq_arr, u_ctrl_arr, u_dist_arr


def simulate(cfg, K, t_end=10.0, dt=0.001, impulse=0.0, disturbance=None,
             gain_scheduler=None):
    """Run a closed-loop simulation of the cart + triple pendulum.

    Parameters
    ----------
    cfg : SystemConfig
    K : ndarray (1, 8)
        LQR gain matrix (used when gain_scheduler is None).
    t_end : float
    dt : float
    impulse : float
    disturbance : ndarray or None
    gain_scheduler : GainScheduler or None
        If provided, uses gain-scheduled control instead of fixed K.
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium
    K_flat = K.flatten()

    N = int(np.ceil(t_end / dt)) + 1
    t_arr = np.linspace(0.0, t_end, N)

    if disturbance is not None:
        assert len(disturbance) >= N, (
            f"Disturbance array length {len(disturbance)} must be >= N={N}"
        )
        dist_arr = disturbance[:N].copy()
    else:
        dist_arr = np.empty(0)

    # Initial conditions
    q0 = q_eq.copy()
    dq0 = np.zeros(4)
    if impulse != 0.0:
        dq0 = apply_impulse(q_eq, p, impulse)

    if gain_scheduler is not None:
        gs_dev, gs_K = gain_scheduler.pack_for_njit()

        # Warmup JIT
        if N > 3:
            _run_loop_gs(3, dt, q0, dq0, q_eq, p,
                         dist_arr[:3] if dist_arr.shape[0] >= 3 else np.empty(0),
                         gs_dev, gs_K)

        q_arr, dq_arr, u_ctrl_arr, u_dist_arr = _run_loop_gs(
            N, dt, q0, dq0, q_eq, p, dist_arr, gs_dev, gs_K
        )
    else:
        # Warmup JIT (first call compiles)
        if N > 3:
            _run_loop(3, dt, q0, dq0, q_eq, K_flat, p, dist_arr[:3] if dist_arr.shape[0] >= 3 else np.empty(0))

        q_arr, dq_arr, u_ctrl_arr, u_dist_arr = _run_loop(
            N, dt, q0, dq0, q_eq, K_flat, p, dist_arr
        )

    return t_arr, q_arr, dq_arr, u_ctrl_arr, u_dist_arr
