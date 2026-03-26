"""Region of Attraction (ROA) estimation via Monte Carlo simulation.

Uses a single @njit function with parallel prange for maximum speed.
The legacy _run_loop is used here because parallel=True compilation of the
monolithic fast loop is prohibitively slow. Since ROA runs many short
simulations, the per-call overhead is less critical than compilation time.
"""

import numpy as np
from numba import njit, prange
from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast, rk4_step_fast


@njit(cache=True)
def _angle_wrap(dx):
    while dx > np.pi:
        dx -= 2.0 * np.pi
    while dx < -np.pi:
        dx += 2.0 * np.pi
    return dx


@njit(cache=True, parallel=True)
def _roa_batch(n_samples, N, dt, q_eq, K_flat, p,
               theta1_devs, theta2_devs, theta3_devs, conv_threshold,
               u_max=1e30, cart_limit=2.0, angle_limit=1.5708):
    """Run ROA simulations in parallel using fast scalar dynamics."""
    converged = np.zeros(n_samples, dtype=np.bool_)
    eq0 = q_eq[0]; eq1 = q_eq[1]; eq2 = q_eq[2]; eq3 = q_eq[3]

    for i in prange(n_samples):
        sq0 = q_eq[0]
        sq1 = q_eq[1] + theta1_devs[i]
        sq2 = q_eq[2] + theta2_devs[i]
        sq3 = q_eq[3] + theta3_devs[i]
        sdq0 = 0.0; sdq1 = 0.0; sdq2 = 0.0; sdq3 = 0.0

        failed = False
        for k in range(N - 1):
            # Control law (inline, scalar)
            z0 = sq0 - eq0
            z1 = _angle_wrap(sq1 - eq1)
            z2 = _angle_wrap(sq2 - eq2)
            z3 = _angle_wrap(sq3 - eq3)
            u_c = -(K_flat[0]*z0 + K_flat[1]*z1 + K_flat[2]*z2 + K_flat[3]*z3
                   + K_flat[4]*sdq0 + K_flat[5]*sdq1 + K_flat[6]*sdq2 + K_flat[7]*sdq3)
            if u_c > u_max:
                u_c = u_max
            elif u_c < -u_max:
                u_c = -u_max

            # RK4 step (fast scalar)
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
                sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u_c, p, dt)

            # Boundedness check
            if sq0 != sq0:  # NaN check
                failed = True
                break
            if abs(sq0 - eq0) > cart_limit:
                failed = True
                break
            if abs(_angle_wrap(sq1 - eq1)) > angle_limit:
                failed = True
                break
            if abs(_angle_wrap(sq2 - eq2)) > angle_limit:
                failed = True
                break
            if abs(_angle_wrap(sq3 - eq3)) > angle_limit:
                failed = True
                break

        if failed:
            continue

        # Convergence check
        max_dev = abs(_angle_wrap(sq1 - eq1))
        d2 = abs(_angle_wrap(sq2 - eq2))
        d3 = abs(_angle_wrap(sq3 - eq3))
        if d2 > max_dev:
            max_dev = d2
        if d3 > max_dev:
            max_dev = d3
        if max_dev < conv_threshold:
            converged[i] = True

    return converged


def _halton_sequence(n_samples, dim, seed=0):
    """Generate Halton quasi-random sequence in [0,1]^dim.

    Low-discrepancy sequence with O(log(N)^d / N) convergence
    vs O(1/sqrt(N)) for pseudo-random — requires ~30% fewer samples.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
    result = np.empty((n_samples, dim))
    for d in range(dim):
        base = primes[d]
        for i in range(n_samples):
            n = i + 1 + seed
            f = 1.0
            r = 0.0
            while n > 0:
                f /= base
                r += f * (n % base)
                n //= base
            result[i, d] = r
    return result


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

    all_theta1 = np.empty(max_samples)
    all_theta2 = np.empty(max_samples)
    all_theta3 = np.empty(max_samples)
    all_converged = np.empty(max_samples, dtype=np.bool_)

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

        halton = _halton_sequence(current_batch, 3, seed=total_samples)
        theta1_devs = (halton[:, 0] * 2 - 1) * max_angle_rad
        theta2_devs = (halton[:, 1] * 2 - 1) * max_angle_rad / 2
        theta3_devs = (halton[:, 2] * 2 - 1) * max_angle_rad / 3

        converged = _roa_batch(current_batch, N, dt, q_eq, K_flat, p,
                               theta1_devs, theta2_devs, theta3_devs,
                               conv_threshold)

        all_theta1[total_samples:total_samples + current_batch] = theta1_devs
        all_theta2[total_samples:total_samples + current_batch] = theta2_devs
        all_theta3[total_samples:total_samples + current_batch] = theta3_devs
        all_converged[total_samples:total_samples + current_batch] = converged
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

    all_theta1 = all_theta1[:total_samples]
    all_theta2 = all_theta2[:total_samples]
    all_theta3 = all_theta3[:total_samples]
    all_converged = all_converged[:total_samples]

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
