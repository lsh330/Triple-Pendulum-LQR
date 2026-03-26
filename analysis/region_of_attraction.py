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
                 t_horizon=5.0, seed=42, convergence_ci_width=0.05,
                 max_samples=2000, batch_size=200):
    """Estimate the region of attraction with adaptive sample sizing.

    Runs batches of Monte Carlo simulations until the 95% confidence
    interval for the success rate is narrower than convergence_ci_width,
    or max_samples is reached.
    """
    rng = np.random.RandomState(seed)
    q_eq = cfg.equilibrium
    p = cfg.pack()
    K_flat = K.flatten()

    max_angle_rad = np.deg2rad(max_angle_deg)
    N = int(np.ceil(t_horizon / dt)) + 1
    conv_threshold = np.deg2rad(1.0)

    all_theta1 = np.empty(0)
    all_theta2 = np.empty(0)
    all_theta3 = np.empty(0)
    all_converged = np.empty(0, dtype=np.bool_)

    # Warmup
    warmup_t1 = rng.uniform(-max_angle_rad, max_angle_rad, 2)
    warmup_t2 = rng.uniform(-max_angle_rad / 2, max_angle_rad / 2, 2)
    warmup_t3 = rng.uniform(-max_angle_rad / 3, max_angle_rad / 3, 2)
    _roa_batch(2, min(N, 10), dt, q_eq, K_flat, p,
               warmup_t1, warmup_t2, warmup_t3, conv_threshold)

    total_samples = 0
    while total_samples < max_samples:
        current_batch = min(batch_size if total_samples > 0 else n_samples,
                           max_samples - total_samples)

        theta1_devs = rng.uniform(-max_angle_rad, max_angle_rad, current_batch)
        theta2_devs = rng.uniform(-max_angle_rad / 2, max_angle_rad / 2, current_batch)
        theta3_devs = rng.uniform(-max_angle_rad / 3, max_angle_rad / 3, current_batch)

        converged = _roa_batch(current_batch, N, dt, q_eq, K_flat, p,
                               theta1_devs, theta2_devs, theta3_devs,
                               conv_threshold)

        all_theta1 = np.concatenate([all_theta1, theta1_devs])
        all_theta2 = np.concatenate([all_theta2, theta2_devs])
        all_theta3 = np.concatenate([all_theta3, theta3_devs])
        all_converged = np.concatenate([all_converged, converged])
        total_samples += current_batch

        # Check convergence of estimate
        rate = np.mean(all_converged)
        # Wilson score 95% CI width
        z = 1.96
        n = total_samples
        denom = 1 + z*z/n
        center = (rate + z*z/(2*n)) / denom
        margin = z * np.sqrt((rate*(1-rate) + z*z/(4*n)) / n) / denom
        ci_width = 2 * margin

        if total_samples >= n_samples and ci_width < convergence_ci_width:
            break

    theta1_devs_deg = np.rad2deg(all_theta1)
    theta2_devs_deg = np.rad2deg(all_theta2)
    success_rate = np.mean(all_converged)
    max_stable = np.max(np.abs(theta1_devs_deg[all_converged])) if np.any(all_converged) else 0.0

    initial_conditions = np.zeros((total_samples, 4))
    initial_conditions[:, 0] = 0.0
    initial_conditions[:, 1] = q_eq[1] + all_theta1
    initial_conditions[:, 2] = q_eq[2] + all_theta2
    initial_conditions[:, 3] = q_eq[3] + all_theta3

    return {
        'initial_conditions': initial_conditions,
        'converged': all_converged,
        'success_rate': float(success_rate),
        'max_stable_deviation_deg': float(max_stable),
        'theta1_devs': theta1_devs_deg,
        'theta2_devs': theta2_devs_deg,
        'converged_mask': all_converged,
        'total_samples': total_samples,
        'ci_width': float(ci_width) if total_samples > 0 else 1.0,
    }
