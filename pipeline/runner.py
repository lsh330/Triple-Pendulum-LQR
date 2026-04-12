"""Orchestrator: config -> LQR -> simulation -> analysis -> visualization -> save."""

import numpy as np
import matplotlib.pyplot as plt

from utils.logger import get_logger
from pipeline.defaults import (T_END, DT, IMPULSE, DIST_AMPLITUDE,
                                DIST_BANDWIDTH, SEED, U_MAX,
                                USE_ILQR, ILQR_HORIZON, ILQR_ITERATIONS,
                                GAIN_SCHEDULER, ADAPTIVE_Q)
from pipeline.save_outputs import save_figure, save_animation
from control.lqr import compute_lqr_gains
from control.ilqr import compute_ilqr_gains
from control.closed_loop import compute_closed_loop
from control.comparison import compare_controllers
from simulation.disturbance.generate_disturbance import generate_disturbance
from simulation.loop.time_loop import simulate
from analysis.state.derived_state import compute_derived_state
from analysis.energy.total_energy import compute_energy
from analysis.frequency.frequency_response import compute_frequency_response
from analysis.lqr_verification.compute_verification import compute_lqr_verification, compute_monte_carlo_robustness
from analysis.summary.print_summary import print_summary
from visualization.animation.show_animation import show_animation
from visualization.dynamics_plots.show_dynamics_plots import show_dynamics_plots
from visualization.control_plots.show_control_plots import show_control_plots
from visualization.lqr_plots.show_lqr_plots import show_lqr_plots
from visualization.comparison_plots.show_comparison_plots import show_comparison_plots
from analysis.region_of_attraction import estimate_roa
from analysis.gain_scheduling_stability import verify_gain_scheduling_stability
from control.gain_scheduling import GainScheduler, MultiAxisGainScheduler
from visualization.roa_plots.show_roa_plots import show_roa_plots
from simulation.warmup import warmup_jit


log = get_logger()


def run(cfg, t_end=T_END, dt=DT, impulse=IMPULSE,
        dist_amplitude=DIST_AMPLITUDE, dist_bandwidth=DIST_BANDWIDTH, seed=SEED,
        u_max=U_MAX, use_ilqr=USE_ILQR, ilqr_horizon=ILQR_HORIZON,
        ilqr_iterations=ILQR_ITERATIONS, gain_scheduler_type=GAIN_SCHEDULER,
        adaptive_q=ADAPTIVE_Q, no_display=False):
    """Run the full simulation pipeline.

    Parameters
    ----------
    gain_scheduler_type : str
        "1d" for cubic Hermite on theta1, "3d" for trilinear on (theta1,2,3).
    adaptive_q : bool
        If True, use inertia-scaled Q matrix (Bryson's rule) instead of fixed.
    """

    q_eq = cfg.equilibrium

    # 0. Pre-trigger all JIT compilations
    log.info("Warming up JIT-compiled functions...")
    warmup_jit()

    # 1. LQR with optional adaptive Q
    Q_mat = None
    if adaptive_q:
        from control.cost_matrices.default_Q import adaptive_Q
        Q_mat = adaptive_Q(cfg)
        log.info("Using adaptive Q matrix (Bryson's rule, inertia-scaled)")

    log.info("Computing LQR gains...")
    K, A, B, P, Q, R = compute_lqr_gains(cfg, Q=Q_mat)
    cl = compute_closed_loop(A, B, K)
    log.info("K = %s", np.array2string(K.flatten(), precision=2))
    log.info("All CL poles stable: %s", cl['is_stable'])

    # 1b. Gain scheduling
    if gain_scheduler_type == "3d":
        log.info("Building 3D multi-axis gain scheduler (7x5x5 = 175 points)...")
        gs = MultiAxisGainScheduler(cfg)
        log.info("  Note: fast loop uses 1D cubic Hermite projection (theta2=theta3=0 slice)")
    else:
        log.info("Building 1D gain scheduler (cubic Hermite, 7 points)...")
        gs = GainScheduler(cfg)

    # 1c. iLQR (optional trajectory optimization)
    ilqr_data = None
    if use_ilqr:
        log.info("Computing iLQR trajectory optimization...")
        q0_ilqr = q_eq.copy()
        dq0_ilqr = np.zeros(4)
        if impulse != 0.0:
            from simulation.initial_conditions.impulse_response import apply_impulse
            dq0_ilqr = apply_impulse(q_eq, cfg.pack(), impulse)
        K_traj, x_traj = compute_ilqr_gains(
            cfg, q0_ilqr, dq0_ilqr,
            N_horizon=ilqr_horizon, dt=dt, n_iter=ilqr_iterations
        )
        log.info("  iLQR trajectory optimized: horizon=%d steps", ilqr_horizon)
        ilqr_data = {"K_traj": K_traj, "x_traj": x_traj}

    # 1d. Control method comparison
    log.info("Computing control method comparison...")
    comparison_data = compare_controllers(A, B, K)

    # 2. Disturbance
    log.info("Generating disturbance...")
    t_tmp = np.linspace(0, t_end, int(t_end / dt) + 1)
    dist = generate_disturbance(t_tmp, amplitude=dist_amplitude,
                                bandwidth=dist_bandwidth, seed=seed)

    # 3. Simulate with gain-scheduled control
    log.info("Simulating (gain-scheduled, %s)...", gain_scheduler_type)
    t, q, dq, u_ctrl, u_dist, u_raw_peak, n_sat = simulate(cfg, K, t_end=t_end, dt=dt,
                                                            impulse=impulse, disturbance=dist,
                                                            gain_scheduler=gs, u_max=u_max)
    log.info("Done: %d steps", len(t))

    # 4. Analysis
    state = compute_derived_state(cfg, q, dq)
    energy = compute_energy(cfg, q, dq,
                            state["phi1"], state["phi2"], state["phi3"])
    A_cl = cl["A_cl"]
    freq_data = compute_frequency_response(A, B, K, A_cl)
    lqr_verif = compute_lqr_verification(t, q, dq, q_eq, K, A, B, P, Q, R)
    log.info("Computing Monte Carlo robustness...")
    mc_robustness = compute_monte_carlo_robustness(cfg)
    print_summary(q, dq, state, u_ctrl, u_dist, freq_data,
                  u_max=u_max, u_raw_peak=u_raw_peak, n_saturated=n_sat)

    # 4b. ROA and Gain Scheduling stability
    log.info("Estimating Region of Attraction...")
    roa_data = estimate_roa(cfg, K, n_samples=500, max_angle_deg=45, t_horizon=5.0)
    log.info("  ROA success rate: %.0f%% (%d samples)",
             roa_data['success_rate']*100, roa_data.get('total_samples', 500))
    log.info("  Max stable deviation: %.1f deg", roa_data['max_stable_deviation_deg'])

    log.info("Verifying gain scheduling stability...")
    gs_1d = gs if isinstance(gs, GainScheduler) else GainScheduler(cfg)
    if not isinstance(gs, GainScheduler):
        log.info("  (3D scheduler: stability verified using 1D proxy)")
    gs_stability = verify_gain_scheduling_stability(cfg, gs_1d)
    log.info("  All operating points stable: %s", gs_stability['all_points_stable'])
    log.info("  Interpolated all stable: %s", gs_stability['interpolated_all_stable'])
    log.info("  Max Re(eigenvalue): %.4f", gs_stability['max_eigenvalue_real'])

    # 5. Visualization
    fig_anim, ani = show_animation(cfg, t, state, dt=dt)
    fig_dyn = show_dynamics_plots(t, q, dq, state, energy, u_ctrl=u_ctrl)
    fig_ctrl = show_control_plots(t, u_ctrl, u_dist, dt, freq_data)
    fig_lqr = show_lqr_plots(lqr_verif, mc_robustness=mc_robustness)
    fig_roa = show_roa_plots(roa_data, gs_stability)
    fig_cmp = show_comparison_plots(comparison_data)

    # 6. Save outputs
    log.info("Saving outputs...")
    save_figure(fig_dyn, "dynamics_analysis")
    save_figure(fig_ctrl, "control_analysis")
    save_figure(fig_lqr, "lqr_verification")
    save_figure(fig_roa, "roa_analysis")
    save_figure(fig_cmp, "comparison_analysis")
    save_animation(ani, "animation", fps=30)

    if not no_display:
        plt.show()
    return ani


def run_switching(cfg, source="DDD", target="UUU", t_end=30.0, dt=0.001,
                  k_energy=50.0, u_max=200.0, no_display=False):
    """Run the form-switching simulation pipeline.

    Orchestration order:

    1. Warm up all JIT-compiled functions.
    2. Plan the minimal-hop transition path (BFS on Hamming graph).
    3. Build :class:`~control.supervisor.form_switch_supervisor.FormSwitchSupervisor`
       — computes LQR gains and Lyapunov ROA thresholds for every target
       configuration in the path.
    4. Pack supervisor data into flat numpy arrays for the JIT loop.
    5. Run the switching simulation via :func:`~simulation.loop.time_loop.simulate_switching`.
    6. Analyse and visualise results.
    7. Save figures and animation to ``images/``.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration.
    source : str
        Starting equilibrium configuration (default "DDD").
    target : str
        Goal equilibrium configuration (default "UUU").
    t_end : float
        Total simulation time (seconds).
    dt : float
        Integration timestep (seconds).
    k_energy : float
        Energy shaping gain for the swing-up controller.
    u_max : float
        Actuator saturation limit (N).
    no_display : bool
        If True, skip ``plt.show()``.

    Returns
    -------
    dict
        Keys: ``t``, ``q``, ``dq``, ``u``, ``mode``, ``stage``, ``energy``,
        ``path``, ``state``, ``animation``.
    """
    from control.supervisor.form_switch_supervisor import FormSwitchSupervisor
    from control.supervisor.transition_graph import plan_transition
    from simulation.loop.time_loop import simulate_switching

    # 0. Warmup JIT
    log.info("Warming up JIT-compiled functions...")
    warmup_jit()

    # 1. Plan transition path
    path = plan_transition(source, target)
    log.info("Transition path: %s", " -> ".join(path))

    # 2. Build supervisor (LQR gains + ROA estimation for each stage target)
    log.info("Building form-switch supervisor (LQR + ROA for %d stages)...",
             len(path) - 1)
    supervisor = FormSwitchSupervisor(cfg, k_energy=k_energy, u_max=u_max)

    for name in path[1:]:
        roa = supervisor._roa_data.get(name)
        if roa is not None:
            log.info("  %s ROA: rho=%.3f, rho_in=%.3f, success=%.0f%% (%d/%d)",
                     name, roa["rho"], roa["rho_in"],
                     roa["success_rate"] * 100,
                     roa["n_converged"], roa["n_total"])
        else:
            log.warning("  %s: ROA estimation unavailable (using defaults)", name)

    # 3. Pack for @njit loop
    sv_data = supervisor.pack_for_njit(path)

    # 4. Simulate
    log.info("Running switching simulation (%.1f s, dt=%.4f)...", t_end, dt)
    t, q, dq, u, mode, stage, energy = simulate_switching(
        cfg, path, sv_data, t_end=t_end, dt=dt
    )
    log.info("Simulation complete: %d steps", len(t))

    nan_mask = np.isnan(q[:, 0])
    if np.any(nan_mask):
        first_nan = int(np.where(nan_mask)[0][0])
        log.warning("Switching simulation diverged at step %d / %d.", first_nan, len(t))

    # 5. Analysis
    from analysis.state.derived_state import compute_derived_state
    state = compute_derived_state(cfg, q, dq)

    # Energy dict for show_dynamics_plots (KE/PE not separately computed here)
    energy_dict = {
        "KE": np.zeros(len(t)),
        "PE": np.zeros(len(t)),
        "TE": energy,
    }

    # 6. Visualisation
    fig_anim, ani = show_animation(cfg, t, state, dt=dt)
    fig_dyn = show_dynamics_plots(t, q, dq, state, energy_dict, u_ctrl=u)

    # 7. Save
    log.info("Saving outputs...")
    save_figure(fig_dyn, "switching_dynamics")
    save_animation(ani, "switching_animation", fps=30)

    if not no_display:
        plt.show()

    return {
        "t": t,
        "q": q,
        "dq": dq,
        "u": u,
        "mode": mode,
        "stage": stage,
        "energy": energy,
        "path": path,
        "state": state,
        "animation": ani,
    }
