"""Main simulation loop -- fully JIT-compiled for maximum performance."""

import numpy as np
from numba import njit
from simulation.integrator.rk4_step import rk4_step
from simulation.loop.control_law import compute_control
from simulation.initial_conditions.impulse_response import apply_impulse
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def _run_loop(N, dt, q0, dq0, q_eq, K_flat, p, disturbance, u_max=1e30):
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
        if u_c > u_max:
            u_c = u_max
        elif u_c < -u_max:
            u_c = -u_max
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
    u_c = compute_control(q, dq, q_eq, K_flat)
    if u_c > u_max:
        u_c = u_max
    elif u_c < -u_max:
        u_c = -u_max
    u_ctrl_arr[N - 1] = u_c
    if has_dist and disturbance.shape[0] >= N:
        u_dist_arr[N - 1] = disturbance[N - 1]
    else:
        u_dist_arr[N - 1] = 0.0

    return q_arr, dq_arr, u_ctrl_arr, u_dist_arr


def simulate(cfg, K, t_end=10.0, dt=0.001, impulse=0.0, disturbance=None,
             gain_scheduler=None, u_max=1e30):
    """Run a closed-loop simulation of the cart + triple pendulum.

    Uses zero-allocation scalar-state RK4 loop for maximum performance.

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
    u_max : float
        Symmetric actuator force saturation limit (N). Default 1e30 (no limit).
    """
    from simulation.loop.time_loop_fast import _run_loop_fast, _run_loop_gs_fast

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
        gs_dev, gs_K, gs_slopes = gain_scheduler.pack_for_njit()

        # Warmup JIT
        if N > 3:
            _run_loop_gs_fast(3, dt, q0, dq0, q_eq, p,
                              dist_arr[:3] if dist_arr.shape[0] >= 3 else np.empty(0),
                              gs_dev, gs_K, gs_slopes, u_max)

        q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_sat = _run_loop_gs_fast(
            N, dt, q0, dq0, q_eq, p, dist_arr, gs_dev, gs_K, gs_slopes, u_max
        )
    else:
        # Warmup JIT (first call compiles)
        if N > 3:
            _run_loop_fast(3, dt, q0, dq0, q_eq, K_flat, p,
                           dist_arr[:3] if dist_arr.shape[0] >= 3 else np.empty(0), u_max)

        q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, n_sat = _run_loop_fast(
            N, dt, q0, dq0, q_eq, K_flat, p, dist_arr, u_max
        )

    # Post-simulation validation
    from utils.logger import get_logger
    _log = get_logger()
    if np.any(np.isnan(q_arr)):
        nan_step = np.where(np.isnan(q_arr[:, 0]))[0]
        first_nan = nan_step[0] if len(nan_step) > 0 else -1
        _log.warning("Simulation diverged (NaN detected at step %d / %d). "
                     "Possible causes: mass matrix singularity, dt too large, "
                     "or unstable controller.", first_nan, N)

    return t_arr, q_arr, dq_arr, u_ctrl_arr, u_dist_arr, u_raw_peak, int(n_sat)


def simulate_switching(cfg, transition_path, supervisor_data,
                       t_end=30.0, dt=0.001):
    """Run a form-switching simulation across multiple equilibrium configurations.

    Calls the zero-allocation ``@njit`` switching loop which implements a
    three-mode FSM (swing-up / LQR-catch / stabilised) per transition stage.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration.
    transition_path : list of str
        Ordered list of equilibrium configuration names, e.g.
        ``["DDD", "UDD", "UUD", "UUU"]``.  The first entry is the
        starting configuration; subsequent entries are successive targets.
    supervisor_data : dict
        Output of :meth:`FormSwitchSupervisor.pack_for_njit`.
    t_end : float
        Total simulation duration (seconds).
    dt : float
        Integration timestep (seconds).

    Returns
    -------
    t : (N,) float64
        Time vector.
    q : (N, 4) float64
        Configuration trajectory [x, theta1, theta2, theta3].
    dq : (N, 4) float64
        Velocity trajectory.
    u : (N,) float64
        Applied control force trajectory (N).
    mode : (N,) int32
        FSM mode at each step (0=swing-up, 1=LQR-catch, 2=stabilised, -1=NaN).
    stage : (N,) int32
        Active transition stage at each step.
    energy : (N,) float64
        Physical total energy at each step (J).
    """
    from simulation.loop.time_loop_switching import _run_loop_switching
    from parameters.equilibrium import equilibrium

    N = int(t_end / dt) + 1
    t = np.linspace(0.0, t_end, N)

    # Start at rest at the source equilibrium
    q0 = equilibrium(transition_path[0])
    dq0 = np.zeros(4)
    p = cfg.pack()

    q, dq, u, mode, stage, energy = _run_loop_switching(
        N, dt,
        q0[0], q0[1], q0[2], q0[3],
        dq0[0], dq0[1], dq0[2], dq0[3],
        p,
        supervisor_data["n_stages"],
        supervisor_data["all_q_eq"],
        supervisor_data["all_K_flat"],
        supervisor_data["all_P_flat"],
        supervisor_data["all_E_target"],
        supervisor_data["all_rho_in"],
        supervisor_data["all_rho_out"],
        supervisor_data["k_energy"],
        supervisor_data["u_max"],
    )

    return t, q, dq, u, mode, stage, energy
