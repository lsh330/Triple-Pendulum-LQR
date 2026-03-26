"""Animate the cart + triple inverted pendulum."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from visualization.common.colors import LINK_COLORS, link_labels


def show_animation(cfg, t, state, dt=0.002):
    """Create and return (fig, ani) for the cart-pendulum animation.

    Parameters
    ----------
    cfg : system configuration with mc, L1, L2, L3, etc.
    t : 1-D time array
    state : dict from compute_derived_state (needs cart_x, P1x, P1y, ...)
    dt : simulation time step used to compute frame skip for 30 fps
    """
    cart_x = state["cart_x"]
    P1x, P1y = state["P1x"], state["P1y"]
    P2x, P2y = state["P2x"], state["P2y"]
    P3x, P3y = state["P3x"], state["P3y"]

    # Frame skip for ~30 fps
    fps = 30
    skip = max(1, int(1.0 / (fps * dt)))
    idx = np.arange(0, len(t), skip)

    fig, ax = plt.subplots(figsize=(10, 7))
    total_L = cfg.L1 + cfg.L2 + cfg.L3
    margin = total_L * 0.3
    ax.set_xlim(cart_x.min() - total_L - margin, cart_x.max() + total_L + margin)
    ax.set_ylim(-total_L - margin, total_L + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Cart + Triple Inverted Pendulum")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="brown", lw=2, zorder=0)  # ground line

    # Cart
    cart_w, cart_h = 0.3, 0.15
    cart_rect = Rectangle((0, 0), cart_w, cart_h, fc="0.6", ec="k", zorder=5)
    ax.add_patch(cart_rect)
    cart_label = ax.text(0, 0, f"mc={cfg.mc:.2f}kg", fontsize=7,
                         ha="center", va="top", zorder=6)

    # Links
    labels = link_labels(cfg)
    lines = []
    for c, lbl in zip(LINK_COLORS, labels):
        (line,) = ax.plot([], [], color=c, lw=3, solid_capstyle="round",
                          label=lbl, zorder=4)
        lines.append(line)

    # Tip trace
    (trace,) = ax.plot([], [], color="tomato", lw=0.8, alpha=0.6, zorder=2)
    trace_x, trace_y = [], []

    # Time text
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top")

    ax.legend(loc="upper right", fontsize=7)

    def _init():
        cart_rect.set_xy((-cart_w / 2, -cart_h / 2))
        for ln in lines:
            ln.set_data([], [])
        trace.set_data([], [])
        time_text.set_text("")
        return [cart_rect, *lines, trace, time_text, cart_label]

    def _update(frame):
        i = idx[frame]
        cx = cart_x[i]

        # Cart rectangle
        cart_rect.set_xy((cx - cart_w / 2, -cart_h / 2))
        cart_label.set_position((cx, -cart_h / 2 - 0.02))

        # Links
        xs = [cx, P1x[i], P2x[i], P3x[i]]
        ys = [0.0, P1y[i], P2y[i], P3y[i]]
        for k, ln in enumerate(lines):
            ln.set_data([xs[k], xs[k + 1]], [ys[k], ys[k + 1]])

        # Tip trace
        trace_x.append(P3x[i])
        trace_y.append(P3y[i])
        trace.set_data(trace_x, trace_y)

        time_text.set_text(f"t = {t[i]:.3f} s")
        return [cart_rect, *lines, trace, time_text, cart_label]

    ani = FuncAnimation(fig, _update, init_func=_init,
                        frames=len(idx), interval=1000 // fps, blit=True)
    return fig, ani
