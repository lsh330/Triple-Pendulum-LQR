"""Orchestrator: config -> LQR -> simulation -> analysis -> visualization -> save."""

import numpy as np
import matplotlib.pyplot as plt

from pipeline.defaults import T_END, DT, IMPULSE, DIST_AMPLITUDE, DIST_BANDWIDTH, SEED
from pipeline.save_outputs import save_figure, save_animation
from control.lqr import compute_lqr_gains
from control.closed_loop import compute_closed_loop
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
from analysis.region_of_attraction import estimate_roa
from analysis.gain_scheduling_stability import verify_gain_scheduling_stability
from control.gain_scheduling import GainScheduler
from visualization.roa_plots.show_roa_plots import show_roa_plots
from simulation.warmup import warmup_jit


def run(cfg, t_end=T_END, dt=DT, impulse=IMPULSE,
        dist_amplitude=DIST_AMPLITUDE, dist_bandwidth=DIST_BANDWIDTH, seed=SEED):
    """Run the full simulation pipeline."""

    # 0. Pre-trigger all JIT compilations
    print("Warming up JIT-compiled functions...")
    warmup_jit()

    # 1. LQR
    print("Computing LQR gains...")
    K, A, B, P, Q, R = compute_lqr_gains(cfg)
    cl = compute_closed_loop(A, B, K)
    print(f"K = {np.array2string(K.flatten(), precision=2)}")
    print(f"All CL poles stable: {cl['is_stable']}")

    # 2. Disturbance
    print("Generating disturbance...")
    t_tmp = np.linspace(0, t_end, int(t_end / dt) + 1)
    dist = generate_disturbance(t_tmp, amplitude=dist_amplitude,
                                bandwidth=dist_bandwidth, seed=seed)

    # 3. Simulate
    print("Simulating...")
    t, q, dq, u_ctrl, u_dist = simulate(cfg, K, t_end=t_end, dt=dt,
                                          impulse=impulse, disturbance=dist)
    print(f"Done: {len(t)} steps")

    # 4. Analysis
    state = compute_derived_state(cfg, q, dq)
    energy = compute_energy(cfg, q, dq,
                            state["phi1"], state["phi2"], state["phi3"])
    A_cl = cl["A_cl"]
    freq_data = compute_frequency_response(A, B, K, A_cl)
    q_eq = cfg.equilibrium
    lqr_verif = compute_lqr_verification(t, q, dq, q_eq, K, A, B, P, Q, R)
    print("Computing Monte Carlo robustness...")
    mc_robustness = compute_monte_carlo_robustness(cfg)
    print_summary(q, dq, state, u_ctrl, u_dist, freq_data)

    # 4b. ROA and Gain Scheduling analysis
    print("Estimating Region of Attraction...")
    roa_data = estimate_roa(cfg, K, n_samples=100, max_angle_deg=30, t_horizon=3.0)
    print(f"  ROA success rate: {roa_data['success_rate']*100:.0f}%")
    print(f"  Max stable deviation: {roa_data['max_stable_deviation_deg']:.1f} deg")

    print("Verifying gain scheduling stability...")
    gs = GainScheduler(cfg)
    gs_stability = verify_gain_scheduling_stability(cfg, gs)
    print(f"  All operating points stable: {gs_stability['all_points_stable']}")
    print(f"  Interpolated all stable: {gs_stability['interpolated_all_stable']}")
    print(f"  Max Re(eigenvalue): {gs_stability['max_eigenvalue_real']:.4f}")

    # 5. Visualization
    fig_anim, ani = show_animation(cfg, t, state, dt=dt)
    fig_dyn = show_dynamics_plots(t, q, dq, state, energy, u_ctrl=u_ctrl)
    fig_ctrl = show_control_plots(t, u_ctrl, u_dist, dt, freq_data)
    fig_lqr = show_lqr_plots(lqr_verif, mc_robustness=mc_robustness)
    fig_roa = show_roa_plots(roa_data, gs_stability)

    # 6. Save outputs
    print("\nSaving outputs...")
    save_figure(fig_dyn, "dynamics_analysis")
    save_figure(fig_ctrl, "control_analysis")
    save_figure(fig_lqr, "lqr_verification")
    save_figure(fig_roa, "roa_analysis")
    save_animation(ani, "animation", fps=30)

    plt.show()
    return ani
