"""ROA and gain scheduling stability visualization (2x2 figure)."""

import numpy as np
import matplotlib.pyplot as plt
from visualization.common.axis_style import apply_grid


def show_roa_plots(roa_data, gs_stability):
    """Create ROA and gain scheduling stability figure (2x2 subplots).

    Parameters
    ----------
    roa_data : dict
        Output from estimate_roa.
    gs_stability : dict
        Output from verify_gain_scheduling_stability.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    fig.suptitle("Region of Attraction & Gain Scheduling Analysis",
                 fontsize=14, fontweight="bold")

    # ---- (0,0) ROA scatter: theta1 vs theta2 ----
    ax = axes[0, 0]
    mask = roa_data['converged_mask']
    t1 = roa_data['theta1_devs']
    t2 = roa_data['theta2_devs']

    ax.scatter(t1[~mask], t2[~mask], c='red', s=15, alpha=0.5,
               label='Diverged', edgecolors='none')
    ax.scatter(t1[mask], t2[mask], c='green', s=15, alpha=0.5,
               label='Converged', edgecolors='none')
    ax.set_xlabel(r'$\theta_1$ deviation [deg]')
    ax.set_ylabel(r'$\theta_2$ deviation [deg]')
    ax.set_title('Region of Attraction (ROA)')
    ax.legend(fontsize=8)
    apply_grid(ax)

    rate = roa_data['success_rate']
    ax.text(0.02, 0.98,
            f"Success rate: {rate*100:.1f}%\n"
            f"Max stable dev: {roa_data['max_stable_deviation_deg']:.1f} deg",
            transform=ax.transAxes, fontsize=8, ha="left", va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (0,1) ROA success rate by theta1 bins ----
    ax = axes[0, 1]
    abs_t1 = np.abs(t1)
    bin_edges = np.linspace(0, np.max(abs_t1) + 1e-6, 7)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_rates = []
    for k in range(len(bin_edges) - 1):
        in_bin = (abs_t1 >= bin_edges[k]) & (abs_t1 < bin_edges[k + 1])
        if np.any(in_bin):
            bin_rates.append(np.mean(mask[in_bin]) * 100)
        else:
            bin_rates.append(0.0)

    colors = ['green' if r > 50 else 'orange' if r > 20 else 'red'
              for r in bin_rates]
    ax.bar(bin_centers, bin_rates, width=(bin_edges[1] - bin_edges[0]) * 0.8,
           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(r'$|\theta_1|$ deviation [deg]')
    ax.set_ylabel('Success rate [%]')
    ax.set_title('Convergence Rate by Initial Deviation')
    ax.set_ylim(0, 105)
    apply_grid(ax)

    # ---- (1,0) Eigenvalue real parts vs operating point ----
    ax = axes[1, 0]
    op_deg = gs_stability['operating_points_deg']
    eig_list = gs_stability['eigenvalues_per_point']

    for i, eigs in enumerate(eig_list):
        x_vals = np.full(len(eigs), op_deg[i])
        ax.scatter(x_vals, eigs.real, c='tab:blue', s=20, alpha=0.6,
                   edgecolors='none')

    ax.axhline(0, color='red', ls='--', lw=1, label='Stability boundary')
    ax.set_xlabel('Operating point deviation [deg]')
    ax.set_ylabel('Re(eigenvalue)')
    ax.set_title('Closed-Loop Eigenvalues (Gain Scheduling)')
    ax.legend(fontsize=8)
    apply_grid(ax)

    stable_label = 'YES' if gs_stability['all_points_stable'] else 'NO'
    interp_label = 'YES' if gs_stability['interpolated_all_stable'] else 'NO'
    ax.text(0.98, 0.98,
            f"All points stable: {stable_label}\n"
            f"Interpolated stable: {interp_label}\n"
            f"Max Re(eig): {gs_stability['max_eigenvalue_real']:.4f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color="green" if gs_stability['all_points_stable'] else "red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (1,1) P matrix condition numbers ----
    ax = axes[1, 1]
    cond = gs_stability['condition_numbers']
    ax.semilogy(op_deg, cond, 'o-', color='tab:purple', lw=1.5, ms=6)
    ax.set_xlabel('Operating point deviation [deg]')
    ax.set_ylabel('Condition number of P')
    ax.set_title('Riccati P Condition Numbers (Robustness)')
    apply_grid(ax)

    ax.text(0.98, 0.98,
            f"Min cond: {np.min(cond):.1f}\nMax cond: {np.max(cond):.1f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    return fig
