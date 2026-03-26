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
from analysis.lqr_verification.compute_verification import compute_lqr_verification
from analysis.summary.print_summary import print_summary
from visualization.animation.show_animation import show_animation
from visualization.dynamics_plots.show_dynamics_plots import show_dynamics_plots
from visualization.control_plots.show_control_plots import show_control_plots
from visualization.lqr_plots.show_lqr_plots import show_lqr_plots


def run(cfg, t_end=T_END, dt=DT, impulse=IMPULSE,
        dist_amplitude=DIST_AMPLITUDE, dist_bandwidth=DIST_BANDWIDTH, seed=SEED):
    """Run the full simulation pipeline."""

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
    print_summary(q, dq, state, u_ctrl, u_dist, freq_data)

    # 5. Visualization
    fig_anim, ani = show_animation(cfg, t, state, dt=dt)
    fig_dyn = show_dynamics_plots(t, q, dq, state, energy)
    fig_ctrl = show_control_plots(t, u_ctrl, u_dist, dt, freq_data)
    fig_lqr = show_lqr_plots(lqr_verif)

    # 6. Save outputs
    print("\nSaving outputs...")
    save_figure(fig_dyn, "dynamics_analysis")
    save_figure(fig_ctrl, "control_analysis")
    save_figure(fig_lqr, "lqr_verification")
    save_animation(ani, "animation", fps=30)

    plt.show()
    return ani
