"""Control method comparison visualization (2x2 figure)."""

import numpy as np
import matplotlib.pyplot as plt
from visualization.common.axis_style import apply_grid, apply_zero_line


def show_comparison_plots(comparison_data):
    """Create a 2x2 figure comparing LQR, PD, and pole placement controllers.

    Parameters
    ----------
    comparison_data : dict
        Output from compare_controllers().

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    fig.suptitle("Control Method Comparison: LQR vs PD vs Pole Placement",
                 fontsize=14, fontweight="bold")

    colors = {"LQR": "tab:blue", "PD": "tab:orange", "Pole Placement": "tab:green"}
    linestyles = {"LQR": "-", "PD": "--", "Pole Placement": "-."}

    # ---- (0,0) Cart position response ----
    ax = axes[0, 0]
    for name, data in comparison_data.items():
        if data["is_stable"]:
            ax.plot(data["t"], data["x"][:, 0], color=colors[name],
                    ls=linestyles[name], lw=1.2, label=name)
        else:
            ax.text(0.5, 0.3, f"{name}: UNSTABLE", transform=ax.transAxes,
                    color=colors[name], fontsize=10, ha="center")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cart position [m]")
    ax.set_title("Impulse Response: Cart Position")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # ---- (0,1) Angle deviation response (theta1) ----
    ax = axes[0, 1]
    for name, data in comparison_data.items():
        if data["is_stable"]:
            ax.plot(data["t"], np.degrees(data["x"][:, 1]),
                    color=colors[name], ls=linestyles[name], lw=1.2, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\theta_1$ deviation [deg]")
    ax.set_title("Impulse Response: Base Link Angle")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # ---- (1,0) Bode magnitude comparison ----
    ax = axes[1, 0]
    for name, data in comparison_data.items():
        ax.semilogx(data["w"], data["L_mag"], color=colors[name],
                    ls=linestyles[name], lw=1.2, label=name)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Open-Loop Bode Magnitude")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # ---- (1,1) Pole map comparison ----
    ax = axes[1, 1]
    for name, data in comparison_data.items():
        eigs = data["eigenvalues"]
        marker = "o" if name == "LQR" else "s" if name == "PD" else "^"
        ax.plot(eigs.real, eigs.imag, marker, color=colors[name],
                ms=7, mfc="none" if name == "LQR" else colors[name],
                mew=1.5, alpha=0.8, label=name)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)

    # Stability summary
    info_lines = []
    for name, data in comparison_data.items():
        status = "Stable" if data["is_stable"] else "UNSTABLE"
        max_re = np.max(data["eigenvalues"].real)
        info_lines.append(f"{name}: {status} (max Re={max_re:.2f})")

    ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Closed-Loop Pole Map")
    ax.legend(fontsize=8)
    apply_grid(ax)

    return fig
