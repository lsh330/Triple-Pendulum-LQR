"""Orchestrator: config -> LQR -> simulation -> analysis -> visualization -> save."""

import numpy as np
import matplotlib.pyplot as plt

from utils.logger import get_logger
from pipeline.defaults import (T_END, DT, IMPULSE, DIST_AMPLITUDE,
                                DIST_BANDWIDTH, SEED, U_MAX,
                                USE_ILQR, ILQR_HORIZON, ILQR_ITERATIONS)
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
from control.gain_scheduling import GainScheduler
from visualization.roa_plots.show_roa_plots import show_roa_plots
from simulation.warmup import warmup_jit


log = get_logger()


def run(cfg, t_end=T_END, dt=DT, impulse=IMPULSE,
        dist_amplitude=DIST_AMPLITUDE, dist_bandwidth=DIST_BANDWIDTH, seed=SEED,
        u_max=U_MAX, use_ilqr=USE_ILQR, ilqr_horizon=ILQR_HORIZON,
        ilqr_iterations=ILQR_ITERATIONS, no_display=False):
    """Run the full simulation pipeline."""

    q_eq = cfg.equilibrium

    # 0. Pre-trigger all JIT compilations
    log.info("Warming up JIT-compiled functions...")
    warmup_jit()

    # 1. LQR
    log.info("Computing LQR gains...")
    K, A, B, P, Q, R = compute_lqr_gains(cfg)
    cl = compute_closed_loop(A, B, K)
    log.info("K = %s", np.array2string(K.flatten(), precision=2))
    log.info("All CL poles stable: %s", cl['is_stable'])

    # 1b. Gain scheduling (used in simulation when enabled)
    log.info("Building gain scheduler...")
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

    # 1d. Control method comparison (PD, Pole Placement vs LQR)
    log.info("Computing control method comparison...")
    comparison_data = compare_controllers(A, B, K)

    # 2. Disturbance
    log.info("Generating disturbance...")
    t_tmp = np.linspace(0, t_end, int(t_end / dt) + 1)
    dist = generate_disturbance(t_tmp, amplitude=dist_amplitude,
                                bandwidth=dist_bandwidth, seed=seed)

    # 3. Simulate with gain-scheduled control
    log.info("Simulating (gain-scheduled)...")
    t, q, dq, u_ctrl, u_dist = simulate(cfg, K, t_end=t_end, dt=dt,
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
    print_summary(q, dq, state, u_ctrl, u_dist, freq_data)

    # 4b. ROA and Gain Scheduling stability
    log.info("Estimating Region of Attraction...")
    roa_data = estimate_roa(cfg, K, n_samples=500, max_angle_deg=45, t_horizon=5.0)
    log.info("  ROA success rate: %.0f%% (%d samples)",
             roa_data['success_rate']*100, roa_data.get('total_samples', 500))
    log.info("  Max stable deviation: %.1f deg", roa_data['max_stable_deviation_deg'])

    log.info("Verifying gain scheduling stability...")
    gs_stability = verify_gain_scheduling_stability(cfg, gs)
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
