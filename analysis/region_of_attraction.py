"""Region of Attraction (ROA) estimation via Monte Carlo simulation."""

import numpy as np
from simulation.loop.time_loop import simulate


def estimate_roa(cfg, K, n_samples=500, max_angle_deg=45, dt=0.001,
                 t_horizon=5.0, seed=42):
    """Estimate the region of attraction of the LQR controller.

    Generates random initial angle deviations from equilibrium and simulates
    the closed-loop system to determine which initial conditions converge.

    Parameters
    ----------
    cfg : SystemConfig
    K : ndarray (1, 8)
        LQR gain matrix.
    n_samples : int
        Number of random initial conditions.
    max_angle_deg : float
        Maximum theta1 deviation in degrees.
    dt : float
        Integration time step.
    t_horizon : float
        Simulation horizon in seconds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'initial_conditions' : (n_samples, 4) array of initial q
        'converged' : (n_samples,) boolean array
        'success_rate' : float
        'max_stable_deviation_deg' : float
        'theta1_devs' : array of theta1 deviations in degrees
        'theta2_devs' : array of theta2 deviations in degrees
        'converged_mask' : boolean mask
    """
    rng = np.random.RandomState(seed)
    q_eq = cfg.equilibrium

    max_angle_rad = np.deg2rad(max_angle_deg)

    # Generate random initial conditions
    theta1_devs_rad = rng.uniform(-max_angle_rad, max_angle_rad, n_samples)
    theta2_devs_rad = rng.uniform(-max_angle_rad / 2, max_angle_rad / 2, n_samples)
    theta3_devs_rad = rng.uniform(-max_angle_rad / 3, max_angle_rad / 3, n_samples)

    initial_conditions = np.zeros((n_samples, 4))
    converged = np.zeros(n_samples, dtype=bool)

    convergence_threshold = np.deg2rad(1.0)  # 1 degree

    for i in range(n_samples):
        # Set initial q: cart at 0, angles deviated from equilibrium
        q0_custom = q_eq.copy()
        q0_custom[0] = 0.0  # cart position fixed
        q0_custom[1] += theta1_devs_rad[i]
        q0_custom[2] += theta2_devs_rad[i]
        q0_custom[3] += theta3_devs_rad[i]

        initial_conditions[i] = q0_custom

        # Simulate: we need to run with this custom initial condition.
        # The simulate function sets q0 = q_eq, so we work around it
        # by using the low-level _run_loop directly.
        from simulation.loop.time_loop import _run_loop

        p = cfg.pack()
        K_flat = K.flatten()
        N = int(np.ceil(t_horizon / dt)) + 1
        dq0 = np.zeros(4)
        dist_arr = np.empty(0)

        q_arr, dq_arr, _, _ = _run_loop(N, dt, q0_custom, dq0, q_eq,
                                         K_flat, p, dist_arr)

        # Check convergence: max angle deviation < 1 degree at the end
        q_final = q_arr[-1]
        angle_devs = np.abs(q_final[1:] - q_eq[1:])
        if np.max(angle_devs) < convergence_threshold:
            converged[i] = True

    theta1_devs_deg = np.rad2deg(theta1_devs_rad)
    theta2_devs_deg = np.rad2deg(theta2_devs_rad)

    success_rate = np.mean(converged)

    # Find max stable theta1 deviation
    if np.any(converged):
        max_stable_deviation_deg = np.max(np.abs(theta1_devs_deg[converged]))
    else:
        max_stable_deviation_deg = 0.0

    return {
        'initial_conditions': initial_conditions,
        'converged': converged,
        'success_rate': float(success_rate),
        'max_stable_deviation_deg': float(max_stable_deviation_deg),
        'theta1_devs': theta1_devs_deg,
        'theta2_devs': theta2_devs_deg,
        'converged_mask': converged,
    }
