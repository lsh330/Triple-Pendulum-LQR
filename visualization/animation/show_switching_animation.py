"""Form-switching animation for the cart + triple inverted pendulum.

Extends the base animation in show_animation.py with:
  - Control mode text overlay (SWING_UP / LQR_CATCH / STABILIZED)
  - Ghost (semi-transparent) target equilibrium posture
  - Current energy and target energy text overlay
  - Cart face-colour changes with mode: red / yellow / green
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from visualization.common.colors import LINK_COLORS, link_labels


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODE_NAMES = {0: "SWING_UP", 1: "LQR_CATCH", 2: "STABILIZED"}
_MODE_CART_FC = {
    0: "#FF6666",   # red
    1: "#FFD966",   # yellow
    2: "#66CC66",   # green
}

# 8 equilibrium configs: absolute angles (phi1, phi2, phi3) in rad
_EQ_ANGLES = {
    "DDD": (0.0,       0.0,       0.0),
    "DDU": (0.0,       0.0,       np.pi),
    "DUD": (0.0,       np.pi,     np.pi),
    "DUU": (0.0,       np.pi,     2 * np.pi),
    "UDD": (np.pi,     np.pi,     np.pi),
    "UDU": (np.pi,     np.pi,     2 * np.pi),
    "UUD": (np.pi,     2 * np.pi, 2 * np.pi),
    "UUU": (np.pi,     2 * np.pi, 3 * np.pi),
}


# ---------------------------------------------------------------------------
# Ghost posture helper
# ---------------------------------------------------------------------------

def _ghost_positions(cx, phi1, phi2, phi3, L1, L2, L3):
    """Return (xs, ys) for 4 points: cart -> J1 -> J2 -> tip.

    Angles phi_i are measured from vertical (upward = pi/2 in standard coords).
    Convention: phi = 0 -> link points down, phi = pi -> link points up.
    """
    # Convert absolute angle to Cartesian offset
    # phi measured from positive-y (upward) axis CCW
    # x-offset = L * sin(phi),  y-offset = L * (-cos(phi))
    j1x = cx        + L1 * np.sin(phi1)
    j1y = 0.0       - L1 * np.cos(phi1)
    j2x = j1x       + L2 * np.sin(phi2)
    j2y = j1y       - L2 * np.cos(phi2)
    tipx = j2x      + L3 * np.sin(phi3)
    tipy = j2y      - L3 * np.cos(phi3)
    xs = [cx, j1x, j2x, tipx]
    ys = [0.0, j1y, j2y, tipy]
    return xs, ys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show_switching_animation(cfg, t, state, mode, stage, energy,
                              transition_path=None, target_energies=None,
                              dt=0.001):
    """Create switching animation with mode indicators.

    Parameters
    ----------
    cfg             : SystemConfig with mc, L1, L2, L3, m1, m2, m3
    t               : (N,) time array [s]
    state           : dict with keys cart_x, P1x, P1y, P2x, P2y, P3x, P3y
    mode            : (N,) int32 array of control modes
    stage           : (N,) int32 array of current stage indices
    energy          : (N,) float array of total energy [J]
    transition_path : list of config name strings (e.g. ['DDD', 'UUU'])
    target_energies : dict {stage_id: E_target} or list indexed by stage
    dt              : simulation timestep [s]  (used to compute frame skip)

    Returns
    -------
    (fig, animation)
    """
    # ---- unpack state ----
    cart_x = np.asarray(state["cart_x"])
    P1x = np.asarray(state["P1x"]); P1y = np.asarray(state["P1y"])
    P2x = np.asarray(state["P2x"]); P2y = np.asarray(state["P2y"])
    P3x = np.asarray(state["P3x"]); P3y = np.asarray(state["P3y"])

    mode   = np.asarray(mode,   dtype=int).ravel()
    stage  = np.asarray(stage,  dtype=int).ravel()
    energy = np.asarray(energy, dtype=float).ravel()

    # ---- frame decimation for ~30 fps ----
    fps  = 30
    skip = max(1, int(1.0 / (fps * dt)))
    idx  = np.arange(0, len(t), skip)

    # ---- figure / axes ----
    fig, ax = plt.subplots(figsize=(12, 8))

    total_L = cfg.L1 + cfg.L2 + cfg.L3
    margin  = total_L * 0.35
    ax.set_xlim(cart_x.min() - total_L - margin,
                cart_x.max() + total_L + margin)
    ax.set_ylim(-total_L - margin, total_L + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Cart + Triple Inverted Pendulum (Form-Switching)", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="saddlebrown", lw=2, zorder=0)

    # ---- cart rectangle ----
    cart_w, cart_h = 0.3, 0.15
    cart_rect = Rectangle((0, 0), cart_w, cart_h,
                           fc=_MODE_CART_FC[0], ec="k", zorder=5)
    ax.add_patch(cart_rect)
    cart_label = ax.text(0, 0, f"mc={cfg.mc:.2f} kg",
                         fontsize=7, ha="center", va="top", zorder=6)

    # ---- pendulum links (solid) ----
    labels = link_labels(cfg)
    lines  = []
    for c, lbl in zip(LINK_COLORS, labels):
        (ln,) = ax.plot([], [], color=c, lw=3, solid_capstyle="round",
                        label=lbl, zorder=4)
        lines.append(ln)

    # ---- tip trace ----
    (trace,) = ax.plot([], [], color="tomato", lw=0.8, alpha=0.6, zorder=2)
    trace_x, trace_y = [], []

    # ---- ghost (target equilibrium) links ----
    ghost_lines = []
    for _ in range(3):
        (gln,) = ax.plot([], [], color="gray", lw=2, ls="--",
                         alpha=0.35, zorder=3, solid_capstyle="round")
        ghost_lines.append(gln)

    # ---- text overlays ----
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        fontsize=11, va="top", fontweight="bold")
    mode_text = ax.text(0.02, 0.90, "", transform=ax.transAxes,
                        fontsize=10, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  alpha=0.85, ec="gray"))
    energy_text = ax.text(0.98, 0.97, "", transform=ax.transAxes,
                          fontsize=9, va="top", ha="right",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    alpha=0.85, ec="gray"))
    stage_text = ax.text(0.98, 0.84, "", transform=ax.transAxes,
                         fontsize=9, va="top", ha="right",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   alpha=0.85, ec="gray"))

    # ---- legend: links + ghost ----
    ghost_proxy = Line2D([0], [0], color="gray", lw=2, ls="--", alpha=0.6,
                         label="Target (ghost)")
    ax.legend(handles=[*[Line2D([0], [0], color=c, lw=3, label=lbl)
                          for c, lbl in zip(LINK_COLORS, labels)],
                        ghost_proxy],
              loc="lower right", fontsize=7, framealpha=0.85)

    # ---- target energy lookup ----
    def _get_target_energy(stage_id):
        if target_energies is None:
            return None
        if isinstance(target_energies, dict):
            return target_energies.get(int(stage_id))
        try:
            return target_energies[int(stage_id)]
        except (IndexError, TypeError):
            return None

    # ---- target equilibrium lookup ----
    def _get_target_config(stage_id):
        if transition_path is None:
            return None
        s = int(stage_id)
        if 0 <= s < len(transition_path):
            return transition_path[s]
        return None

    # ====================================================================
    # Animation callbacks
    # ====================================================================

    def _init():
        cart_rect.set_xy((-cart_w / 2, -cart_h / 2))
        for ln in lines:
            ln.set_data([], [])
        for gln in ghost_lines:
            gln.set_data([], [])
        trace.set_data([], [])
        time_text.set_text("")
        mode_text.set_text("")
        energy_text.set_text("")
        stage_text.set_text("")
        return [cart_rect, *lines, *ghost_lines, trace,
                time_text, mode_text, energy_text, stage_text, cart_label]

    def _update(frame):
        i   = idx[frame]
        cx  = cart_x[i]
        mid = int(mode[i])
        sid = int(stage[i])

        # -- cart --
        cart_rect.set_xy((cx - cart_w / 2, -cart_h / 2))
        cart_rect.set_facecolor(_MODE_CART_FC.get(mid, "0.6"))
        cart_label.set_position((cx, -cart_h / 2 - 0.02))

        # -- links --
        xs = [cx, P1x[i], P2x[i], P3x[i]]
        ys = [0.0, P1y[i], P2y[i], P3y[i]]
        for k, ln in enumerate(lines):
            ln.set_data([xs[k], xs[k + 1]], [ys[k], ys[k + 1]])

        # -- tip trace --
        trace_x.append(P3x[i])
        trace_y.append(P3y[i])
        trace.set_data(trace_x, trace_y)

        # -- ghost posture --
        cfg_name = _get_target_config(sid)
        if cfg_name is not None and cfg_name in _EQ_ANGLES:
            phi1_t, phi2_t, phi3_t = _EQ_ANGLES[cfg_name]
            gxs, gys = _ghost_positions(cx, phi1_t, phi2_t, phi3_t,
                                        cfg.L1, cfg.L2, cfg.L3)
            for k, gln in enumerate(ghost_lines):
                gln.set_data([gxs[k], gxs[k + 1]], [gys[k], gys[k + 1]])
        else:
            for gln in ghost_lines:
                gln.set_data([], [])

        # -- time --
        time_text.set_text(f"t = {t[i]:.3f} s")

        # -- mode overlay --
        mode_name = _MODE_NAMES.get(mid, f"MODE {mid}")
        mode_text.set_text(f"Mode: {mode_name}")
        mode_color = {"SWING_UP": "#CC0000",
                      "LQR_CATCH": "#BB8800",
                      "STABILIZED": "#006600"}.get(mode_name, "k")
        mode_text.set_color(mode_color)

        # -- energy overlay --
        e_now  = energy[i]
        e_tgt  = _get_target_energy(sid)
        if e_tgt is not None:
            energy_text.set_text(
                f"E = {e_now:+.2f} J\nE* = {e_tgt:+.2f} J\n"
                f"dE = {e_now - e_tgt:+.2f} J"
            )
        else:
            energy_text.set_text(f"E = {e_now:+.2f} J")

        # -- stage / target config overlay --
        cfg_str = f" ({cfg_name})" if cfg_name else ""
        stage_text.set_text(f"Stage {sid}{cfg_str}")

        return [cart_rect, *lines, *ghost_lines, trace,
                time_text, mode_text, energy_text, stage_text, cart_label]

    ani = FuncAnimation(fig, _update, init_func=_init,
                        frames=len(idx),
                        interval=1000 // fps,
                        blit=True)
    return fig, ani
