"""Static 4-panel figure for form-switching simulation analysis.

Panels
------
1 (top-left)    : Absolute angles phi1, phi2, phi3 vs time
2 (top-right)   : Total energy vs time with stage target levels
3 (bottom-left) : Control input u vs time
4 (bottom-right): Cart position x vs time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization.common.axis_style import apply_grid, apply_zero_line

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Control mode indices
_MODE_SWING_UP   = 0
_MODE_LQR_CATCH  = 1
_MODE_STABILIZED = 2

_MODE_NAMES  = {0: "SWING_UP", 1: "LQR_CATCH", 2: "STABILIZED"}
_MODE_COLORS = {0: "#FFCCCC",  1: "#FFFACC",   2: "#CCFFCC"}   # light R / Y / G

# Distinct pastel colours for up to 8 stages
_STAGE_COLORS = [
    "#CCE5FF",  # light blue
    "#D4EDDA",  # light green
    "#FFF3CD",  # light yellow
    "#F8D7DA",  # light red
    "#E2CCFF",  # light purple
    "#D1ECF1",  # light teal
    "#FFDDC1",  # light orange
    "#E8E8E8",  # light grey
]

# 8 equilibrium configurations (DDD..UUU): absolute angles in rad
# D = down (0), U = up (pi)
_EQ_NAMES = ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"]
_EQ_ANGLES = {
    "DDD": (0.0,      0.0,      0.0),
    "DDU": (0.0,      0.0,      np.pi),
    "DUD": (0.0,      np.pi,    np.pi),
    "DUU": (0.0,      np.pi,    2 * np.pi),
    "UDD": (np.pi,    np.pi,    np.pi),
    "UDU": (np.pi,    np.pi,    2 * np.pi),
    "UUD": (np.pi,    2 * np.pi, 2 * np.pi),
    "UUU": (np.pi,    2 * np.pi, 3 * np.pi),
}


# ---------------------------------------------------------------------------
# Helper: shade background by mode or stage
# ---------------------------------------------------------------------------

def _shade_background(ax, t, label_array, color_map, alpha=0.15):
    """Fill vertical spans for each contiguous run of label_array values.

    Parameters
    ----------
    ax          : matplotlib Axes
    t           : (N,) time array
    label_array : (N,) integer array
    color_map   : dict {label_value: hex_color}
    alpha       : transparency
    """
    if len(t) == 0:
        return
    prev_val = label_array[0]
    t_start  = t[0]
    for i in range(1, len(t)):
        if label_array[i] != prev_val:
            color = color_map.get(prev_val, "#EEEEEE")
            ax.axvspan(t_start, t[i], color=color, alpha=alpha, lw=0)
            t_start  = t[i]
            prev_val = label_array[i]
    # Final segment
    color = color_map.get(prev_val, "#EEEEEE")
    ax.axvspan(t_start, t[-1], color=color, alpha=alpha, lw=0)


def _mode_legend_patches():
    """Return proxy artists for control-mode background legend."""
    patches = []
    for mode_id, name in _MODE_NAMES.items():
        patches.append(
            mpatches.Patch(color=_MODE_COLORS[mode_id], alpha=0.7, label=name)
        )
    return patches


def _stage_color_map(stage):
    """Return a color map for unique stage values present in *stage*."""
    unique = np.unique(stage)
    return {int(s): _STAGE_COLORS[int(s) % len(_STAGE_COLORS)] for s in unique}


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _panel_angles(ax, t, q, mode, transition_path):
    """Panel 1: absolute angles phi1, phi2, phi3."""
    phi1 = q[:, 1]
    phi2 = q[:, 1] + q[:, 2]
    phi3 = q[:, 1] + q[:, 2] + q[:, 3]

    # Background shading by control mode
    _shade_background(ax, t, mode, _MODE_COLORS, alpha=0.18)

    # Angle traces
    ax.plot(t, np.degrees(phi1), color="tab:red",   lw=1.5, label=r"$\varphi_1$")
    ax.plot(t, np.degrees(phi2), color="tab:green", lw=1.5, label=r"$\varphi_2$")
    ax.plot(t, np.degrees(phi3), color="tab:blue",  lw=1.5, label=r"$\varphi_3$")

    # Target equilibrium lines from transition_path
    if transition_path is not None and len(transition_path) > 0:
        _draw_eq_targets(ax, transition_path)

    ax.set_ylabel("Absolute Angle [deg]", fontsize=12)
    ax.set_title("Absolute Angles vs Time", fontsize=14, fontweight="bold")

    # Extend y-axis to 0..3pi deg nicely
    ax.set_ylim(-30, 570)
    ax.set_yticks(np.arange(0, 571, 90))

    # Mode legend on right side, angle legend on left
    handles, labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=_mode_legend_patches(), loc="upper right",
              fontsize=7, framealpha=0.8, title="Mode")

    apply_grid(ax)
    apply_zero_line(ax)


def _draw_eq_targets(ax, transition_path):
    """Draw horizontal dashed lines for target equilibrium angles."""
    styles = ["--", "-.", ":"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    angle_names = [r"$\varphi_1^*$", r"$\varphi_2^*$", r"$\varphi_3^*$"]
    drawn = set()
    for i, cfg_name in enumerate(transition_path):
        angles = _EQ_ANGLES.get(cfg_name)
        if angles is None:
            continue
        for j, (target_rad, label) in enumerate(zip(angles, angle_names)):
            target_deg = np.degrees(target_rad)
            key = (cfg_name, j)
            if key not in drawn:
                ax.axhline(target_deg,
                           color=colors[j], ls=styles[i % len(styles)],
                           lw=0.8, alpha=0.55,
                           label=f"{cfg_name} {label}={target_deg:.0f}deg")
                drawn.add(key)


def _panel_energy(ax, t, energy, mode, stage, target_energies):
    """Panel 2: total energy with stage shading and target levels."""
    stage_cmap = _stage_color_map(stage)
    _shade_background(ax, t, stage, stage_cmap, alpha=0.20)

    ax.plot(t, energy, color="tab:blue", lw=1.5, label="E(t)")

    # Target energy horizontal lines per stage
    if target_energies is not None:
        unique_stages = np.unique(stage)
        for s in unique_stages:
            if isinstance(target_energies, dict):
                e_tgt = target_energies.get(int(s))
            else:
                try:
                    e_tgt = target_energies[int(s)]
                except (IndexError, TypeError):
                    e_tgt = None
            if e_tgt is not None:
                ax.axhline(e_tgt, color="tab:red", ls="--", lw=1.0, alpha=0.7,
                           label=f"E* stage {int(s)} = {e_tgt:.2f} J")

    ax.set_ylabel("Energy [J]", fontsize=12)
    ax.set_title("Total Energy vs Time", fontsize=14, fontweight="bold")

    # Stage legend patches
    stage_patches = [
        mpatches.Patch(color=stage_cmap.get(int(s), "#EEEEEE"), alpha=0.7,
                       label=f"Stage {int(s)}")
        for s in np.unique(stage)
    ]
    handles, labels_list = ax.get_legend_handles_labels()
    energy_handles = handles
    energy_labels  = labels_list
    leg1 = ax.legend(energy_handles, energy_labels, loc="upper right",
                     fontsize=7, framealpha=0.8)
    ax.add_artist(leg1)
    if stage_patches:
        ax.legend(handles=stage_patches, loc="upper left",
                  fontsize=7, framealpha=0.8, title="Stage")

    apply_grid(ax)


def _panel_control(ax, t, u, mode, u_max):
    """Panel 3: control input with saturation lines."""
    _shade_background(ax, t, mode, _MODE_COLORS, alpha=0.18)

    ax.plot(t, u, color="tab:blue", lw=1.2, label="u(t)")

    # Saturation bounds
    ax.axhline( u_max, color="tab:red", ls="--", lw=1.0, alpha=0.8,
               label=f"+u_max = {u_max:.0f} N")
    ax.axhline(-u_max, color="tab:red", ls="--", lw=1.0, alpha=0.8,
               label=f"-u_max = {-u_max:.0f} N")

    ax.set_ylabel("Force [N]", fontsize=12)
    ax.set_title("Control Input vs Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    apply_grid(ax)
    apply_zero_line(ax)


def _panel_cart(ax, t, q, stage):
    """Panel 4: cart position with stage shading."""
    stage_cmap = _stage_color_map(stage)
    _shade_background(ax, t, stage, stage_cmap, alpha=0.20)

    ax.plot(t, q[:, 0], color="k", lw=1.5, label="x(t)")

    ax.set_ylabel("Cart Position [m]", fontsize=12)
    ax.set_title("Cart Position vs Time", fontsize=14, fontweight="bold")

    # Stage patches
    stage_patches = [
        mpatches.Patch(color=stage_cmap.get(int(s), "#EEEEEE"), alpha=0.7,
                       label=f"Stage {int(s)}")
        for s in np.unique(stage)
    ]
    handles, labels_list = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles, labels_list, loc="upper right",
                     fontsize=8, framealpha=0.8)
    ax.add_artist(leg1)
    if stage_patches:
        ax.legend(handles=stage_patches, loc="upper left",
                  fontsize=7, framealpha=0.8, title="Stage")

    apply_grid(ax)
    apply_zero_line(ax)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show_switching_plots(t, q, dq, u, mode, stage, energy,
                         transition_path=None, target_energies=None,
                         u_max=200.0):
    """Create 4-panel switching analysis figure.

    Parameters
    ----------
    t               : (N,) time array [s]
    q               : (N, 4) generalised coordinates [x, th1, th2, th3]
    dq              : (N, 4) generalised velocities
    u               : (N,) control force [N]
    mode            : (N,) int32 control mode  0=SWING_UP 1=LQR_CATCH 2=STABILIZED
    stage           : (N,) int32 current stage index
    energy          : (N,) total physical energy [J]
    transition_path : list of equilibrium config names, e.g. ['DDD','UUU']
    target_energies : dict {stage_id: E_target} or list indexed by stage
    u_max           : saturation level for control input [N]

    Returns
    -------
    matplotlib.figure.Figure
    """
    t      = np.asarray(t,      dtype=float)
    q      = np.asarray(q,      dtype=float)
    dq     = np.asarray(dq,     dtype=float)
    u      = np.asarray(u,      dtype=float).ravel()
    mode   = np.asarray(mode,   dtype=int).ravel()
    stage  = np.asarray(stage,  dtype=int).ravel()
    energy = np.asarray(energy, dtype=float).ravel()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    fig.suptitle("Form-Switching Simulation Analysis", fontsize=14, fontweight="bold")

    ax_angles  = axes[0, 0]
    ax_energy  = axes[0, 1]
    ax_control = axes[1, 0]
    ax_cart    = axes[1, 1]

    # Shared x-axis label
    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]", fontsize=12)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)

    _panel_angles( ax_angles,  t, q, mode, transition_path)
    _panel_energy( ax_energy,  t, energy, mode, stage, target_energies)
    _panel_control(ax_control, t, u, mode, u_max)
    _panel_cart(   ax_cart,    t, q, stage)

    # Tick font size
    for ax in axes.ravel():
        ax.tick_params(labelsize=10)

    return fig
