"""Static plots for system dynamics (positions, velocities, accelerations, energy)."""

import numpy as np
import matplotlib.pyplot as plt

from visualization.common.colors import LINK_COLORS
from visualization.common.axis_style import apply_grid, apply_zero_line


def show_dynamics_plots(t, q, dq, state, energy, u_ctrl=None):
    """Return a figure with a 5x2 grid of dynamics plots."""
    fig, axes = plt.subplots(5, 2, figsize=(15, 16), tight_layout=True)
    fig.suptitle("System Dynamics", fontsize=14, fontweight="bold")

    link_names = ["th1", "th2", "th3"]
    dt = t[1] - t[0]

    # --- Compute accelerations via numerical differentiation ---
    ddq = np.gradient(dq, dt, axis=0)
    cart_acc = ddq[:, 0]
    ang_acc1 = ddq[:, 1]
    ang_acc2 = ddq[:, 2]
    ang_acc3 = ddq[:, 3]

    # (0,0) Cart position
    ax = axes[0, 0]
    ax.plot(t, q[:, 0], "k")
    ax.set_ylabel("x [m]")
    ax.set_title("Cart Position")
    apply_grid(ax); apply_zero_line(ax)

    # (0,1) Cart velocity
    ax = axes[0, 1]
    ax.plot(t, dq[:, 0], "k")
    ax.set_ylabel("dx/dt [m/s]")
    ax.set_title("Cart Velocity")
    apply_grid(ax); apply_zero_line(ax)

    # (1,0) Angle deviations
    ax = axes[1, 0]
    for i, key in enumerate(["dth1", "dth2", "dth3"]):
        ax.plot(t, np.degrees(state[key]), color=LINK_COLORS[i], label=link_names[i])
    ax.set_ylabel("Deviation [deg]")
    ax.set_title("Angle Deviations from Upright")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (1,1) Angular velocities
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(t, np.degrees(dq[:, i+1]), color=LINK_COLORS[i], label=link_names[i])
    ax.set_ylabel("Angular vel [deg/s]")
    ax.set_title("Angular Velocities")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (2,0) Cart acceleration
    ax = axes[2, 0]
    ax.plot(t, cart_acc, "k", lw=0.8)
    ax.set_ylabel("d2x/dt2 [m/s2]")
    ax.set_title("Cart Acceleration")
    apply_grid(ax); apply_zero_line(ax)

    # (2,1) Angular accelerations
    ax = axes[2, 1]
    ax.plot(t, np.degrees(ang_acc1), color=LINK_COLORS[0], lw=0.8, label="ddth1")
    ax.plot(t, np.degrees(ang_acc2), color=LINK_COLORS[1], lw=0.8, label="ddth2")
    ax.plot(t, np.degrees(ang_acc3), color=LINK_COLORS[2], lw=0.8, label="ddth3")
    ax.set_ylabel("Angular accel [deg/s2]")
    ax.set_title("Angular Accelerations")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (3,0) Energy
    ax = axes[3, 0]
    ax.plot(t, energy["KE"], label="KE", color="tab:orange")
    ax.plot(t, energy["PE"], label="PE", color="tab:cyan")
    ax.plot(t, energy["TE"], label="TE", color="k", ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("Energy")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (3,1) Phase portrait
    ax = axes[3, 1]
    for i, key in enumerate(["dth1", "dth2", "dth3"]):
        ax.plot(np.degrees(state[key]), np.degrees(dq[:, i+1]),
                color=LINK_COLORS[i], lw=0.5, alpha=0.8, label=link_names[i])
    ax.plot(0, 0, "k+", ms=12, mew=2)
    ax.set_xlabel("Angle error [deg]")
    ax.set_ylabel("Angular vel [deg/s]")
    ax.set_title("Phase Portrait (Angles)")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (4,0) Control Force
    ax = axes[4, 0]
    if u_ctrl is not None:
        ax.plot(t, u_ctrl, "tab:blue", lw=0.8)
        i_peak = np.argmax(np.abs(u_ctrl))
        ax.annotate(f"Peak: {u_ctrl[i_peak]:.1f} N",
                    xy=(t[i_peak], u_ctrl[i_peak]), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="tab:blue"),
                    xytext=(t[i_peak] + 0.3, u_ctrl[i_peak] * 0.7))
    else:
        ax.text(0.5, 0.5, "No control data", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Control Force")
    apply_grid(ax); apply_zero_line(ax)

    # (4,1) Joint Reaction Estimation (generalized forces = M * ddq approx)
    ax = axes[4, 1]
    # Use mass-weighted acceleration as proxy for generalized forces
    # ddq already computed above; show generalized force estimate per DOF
    ax.plot(t, cart_acc, "k", lw=0.8, label="Cart (ddx)")
    ax.plot(t, ang_acc1, color=LINK_COLORS[0], lw=0.8, label="Joint 1 (ddth1)")
    ax.plot(t, ang_acc2, color=LINK_COLORS[1], lw=0.8, label="Joint 2 (ddth2)")
    ax.plot(t, ang_acc3, color=LINK_COLORS[2], lw=0.8, label="Joint 3 (ddth3)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Generalized accel [rad/s2]")
    ax.set_title("Joint Reaction Estimation (ddq)")
    ax.legend(fontsize=7)
    apply_grid(ax); apply_zero_line(ax)

    return fig
