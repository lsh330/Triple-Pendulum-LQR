"""Region of Attraction (ROA) estimation via Monte Carlo simulation.

Uses a single @njit function with parallel prange for maximum speed.
The legacy _run_loop is used here because parallel=True compilation of the
monolithic fast loop is prohibitively slow. Since ROA runs many short
simulations, the per-call overhead is less critical than compilation time.
"""

import numpy as np
from numba import njit, prange
from simulation.loop.time_loop import _run_loop


@njit(cache=True, parallel=True)
def _roa_batch(n_samples, N, dt, q_eq, K_flat, p,
               theta1_devs, theta2_devs, theta3_devs, conv_threshold,
               u_max=1e30, cart_limit=2.0, angle_limit=1.5708):
    """Run all ROA simulations in parallel using Numba prange."""
    converged = np.zeros(n_samples, dtype=np.bool_)
    dist_arr = np.empty(0)

    for i in prange(n_samples):
        q0 = q_eq.copy()
        q0[1] += theta1_devs[i]
        q0[2] += theta2_devs[i]
        q0[3] += theta3_devs[i]
        dq0 = np.zeros(4)

        q_arr, dq_arr, _, _ = _run_loop(N, dt, q0, dq0, q_eq, K_flat, p,
                                         dist_arr, u_max)

        # Trajectory boundedness check (early exit on divergence)
        failed = False
        for k in range(N):
            if np.isnan(q_arr[k, 0]):
                failed = True
                break
            if abs(q_arr[k, 0] - q_eq[0]) > cart_limit:
                failed = True
                break
            for j in range(1, 4):
                if np.isnan(q_arr[k, j]):
                    failed = True
                    break
                if abs(q_arr[k, j] - q_eq[j]) > angle_limit:
                    failed = True
                    break
            if failed:
                break

        if failed:
            continue

        # Check convergence at final timestep
        max_dev = 0.0
        for j in range(1, 4):
            dev = abs(q_arr[N - 1, j] - q_eq[j])
            if dev > max_dev:
                max_dev = dev
        if max_dev < conv_threshold:
            converged[i] = True

    return converged


def estimate_roa(cfg, K, n_samples=500, max_angle_deg=45, dt=0.001,
                 t_horizon=5.0, seed=42):
    """Estimate the region of attraction of the LQR controller.

    Uses Numba parallel prange to distribute simulations across CPU cores.
    """
    rng = np.random.RandomState(seed)
    q_eq = cfg.equilibrium
    p = cfg.pack()
    K_flat = K.flatten()

    max_angle_rad = np.deg2rad(max_angle_deg)
    theta1_devs_rad = rng.uniform(-max_angle_rad, max_angle_rad, n_samples)
    theta2_devs_rad = rng.uniform(-max_angle_rad / 2, max_angle_rad / 2, n_samples)
    theta3_devs_rad = rng.uniform(-max_angle_rad / 3, max_angle_rad / 3, n_samples)

    N = int(np.ceil(t_horizon / dt)) + 1
    conv_threshold = np.deg2rad(1.0)

    # Warmup (compile the parallel function with small batch)
    _roa_batch(2, min(N, 10), dt, q_eq, K_flat, p,
               theta1_devs_rad[:2], theta2_devs_rad[:2], theta3_devs_rad[:2],
               conv_threshold)

    converged = _roa_batch(n_samples, N, dt, q_eq, K_flat, p,
                           theta1_devs_rad, theta2_devs_rad, theta3_devs_rad,
                           conv_threshold)

    theta1_devs_deg = np.rad2deg(theta1_devs_rad)
    theta2_devs_deg = np.rad2deg(theta2_devs_rad)
    success_rate = np.mean(converged)
    max_stable = np.max(np.abs(theta1_devs_deg[converged])) if np.any(converged) else 0.0

    initial_conditions = np.zeros((n_samples, 4))
    initial_conditions[:, 0] = 0.0
    initial_conditions[:, 1] = q_eq[1] + theta1_devs_rad
    initial_conditions[:, 2] = q_eq[2] + theta2_devs_rad
    initial_conditions[:, 3] = q_eq[3] + theta3_devs_rad

    return {
        'initial_conditions': initial_conditions,
        'converged': converged,
        'success_rate': float(success_rate),
        'max_stable_deviation_deg': float(max_stable),
        'theta1_devs': theta1_devs_deg,
        'theta2_devs': theta2_devs_deg,
        'converged_mask': converged,
    }
