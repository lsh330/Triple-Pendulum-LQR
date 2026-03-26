"""Window 4: LQR Verification -- evidence that LQR is working correctly."""

import numpy as np
import matplotlib.pyplot as plt
from visualization.common.axis_style import apply_grid, apply_zero_line


def show_lqr_plots(lqr_verif, mc_robustness=None):
    """Create LQR verification figure (4x2 = 8 subplots)."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), tight_layout=True)
    fig.suptitle("LQR Verification", fontsize=14, fontweight="bold")

    t = lqr_verif["t"]

    # ---- (0,0) Lyapunov function V(t) = z'Pz ----
    ax = axes[0, 0]
    V = lqr_verif["lyapunov_V"]
    ax.semilogy(t, V, "tab:blue", lw=1)
    ax.set_ylabel("V(t) = z'Pz")
    ax.set_title("Lyapunov Function (must decrease)")
    apply_grid(ax)
    # Check monotonicity
    dV = np.diff(V)
    n_increase = np.sum(dV > 1e-10)
    total = len(dV)
    pct_decrease = (1 - n_increase / total) * 100
    color = "green" if pct_decrease > 95 else "red"
    ax.text(0.98, 0.98, f"Decreasing: {pct_decrease:.1f}%",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (0,1) Riccati P eigenvalues ----
    ax = axes[0, 1]
    P_eig = lqr_verif["P_eigenvalues"]
    idx = np.arange(len(P_eig))
    colors = ["green" if v > 0 else "red" for v in P_eig]
    ax.bar(idx, P_eig, color=colors, alpha=0.7)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Value")
    ax.set_title("Riccati P Eigenvalues (must all be > 0)")
    ax.set_xticks(idx)
    apply_grid(ax)
    all_pos = np.all(P_eig > 0)
    ax.text(0.98, 0.98,
            f"P positive definite: {'YES' if all_pos else 'NO'}",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            color="green" if all_pos else "red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (1,0) LQR cost breakdown ----
    ax = axes[1, 0]
    ax.semilogy(t, lqr_verif["cost_state"], label="State cost (z'Qz)",
                color="tab:blue", lw=0.8)
    ax.semilogy(t, lqr_verif["cost_control"], label="Control cost (u'Ru)",
                color="tab:red", lw=0.8)
    ax.set_ylabel("Instantaneous cost")
    ax.set_title("LQR Cost Breakdown")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # ---- (1,1) Cumulative cost J(t) ----
    ax = axes[1, 1]
    J = lqr_verif["cost_cumulative"]
    ax.plot(t, J, "tab:blue", lw=1)
    ax.set_ylabel("J(t) = integral(z'Qz + u'Ru)")
    ax.set_title("Cumulative LQR Cost (must converge)")
    apply_grid(ax)
    # Annotate final value
    ax.axhline(J[-1], color="gray", ls=":", lw=0.7)
    ax.text(0.98, 0.5, f"J_final = {J[-1]:.2f}",
            transform=ax.transAxes, fontsize=9, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (2,0) Return difference |1 + L(jw)| (Kalman inequality) ----
    ax = axes[2, 0]
    w_rd = lqr_verif["return_diff_w"]
    rd = lqr_verif["return_difference"]
    ax.semilogx(w_rd, 20 * np.log10(rd), "tab:blue", lw=1)
    ax.axhline(0, color="red", ls="--", lw=1, label="|1+L| = 1 (0 dB)")
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("|1 + L(jw)| [dB]")
    ax.set_title("Return Difference (Kalman: must be >= 0 dB)")
    ax.legend(fontsize=8)
    apply_grid(ax)

    min_rd = np.min(rd)
    min_rd_dB = 20 * np.log10(min_rd)
    i_min = np.argmin(rd)
    kalman_ok = min_rd >= 1.0 - 1e-6
    ax.text(0.98, 0.02,
            f"Min |1+L| = {min_rd_dB:.2f} dB @ {w_rd[i_min]:.1f} rad/s\n"
            f"Kalman inequality: {'SATISFIED' if kalman_ok else 'VIOLATED'}",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="green" if kalman_ok else "red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (2,1) Nyquist verification ----
    ax = axes[2, 1]
    L_jw = lqr_verif["L_jw_full"]
    w_full = lqr_verif["return_diff_w"]

    ax.plot(L_jw.real, L_jw.imag, "tab:blue", lw=0.8, label="L(jw), w>0")
    ax.plot(L_jw.real, -L_jw.imag, "tab:blue", ls="--", lw=0.5, alpha=0.5, label="L(jw), w<0")
    ax.plot(-1, 0, "rx", ms=10, mew=2)

    theta_c = np.linspace(0, 2*np.pi, 200)
    ax.plot(-1 + np.cos(theta_c), np.sin(theta_c), "r--", lw=0.5, alpha=0.3)

    # Direction arrows
    n_pts = len(L_jw)
    for frac in [0.05, 0.2, 0.5]:
        idx = int(frac * n_pts)
        if idx + 1 < n_pts:
            ax.annotate("", xy=(L_jw[idx+1].real, L_jw[idx+1].imag),
                        xytext=(L_jw[idx].real, L_jw[idx].imag),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5))

    n_enc = lqr_verif["n_encirclements_cw"]
    n_unstable = lqr_verif["n_unstable_ol"]
    nyq_ok = lqr_verif["nyquist_criterion_ok"]

    ax.set_xlabel("Real"); ax.set_ylabel("Imag")
    ax.set_title("Nyquist -- Encirclement Verification")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")

    r_max = max(3.0, np.percentile(np.abs(L_jw), 90) * 1.2)
    ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
    apply_grid(ax)

    ax.text(0.98, 0.02,
            f"CW encirclements of (-1,0): {n_enc}\n"
            f"Unstable OL poles: {n_unstable}\n"
            f"Nyquist criterion (N=P): {'PASS' if nyq_ok else 'FAIL'}",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="green" if nyq_ok else "red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # ---- (3,0) Monte Carlo Bode magnitude ----
    ax = axes[3, 0]
    if mc_robustness is not None and len(mc_robustness.get("mc_bode_mag", [])) > 0:
        w_mc = mc_robustness["mc_bode_w"]
        for i, mag_dB in enumerate(mc_robustness["mc_bode_mag"]):
            ax.semilogx(w_mc, mag_dB, color="tab:gray", alpha=0.3, lw=0.5,
                        label="Perturbed" if i == 0 else None)
        ax.semilogx(w_mc, mc_robustness["nominal_bode_mag"],
                    color="tab:blue", lw=1.5, label="Nominal")
        ax.set_xlabel("Frequency [rad/s]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title("Monte Carlo Bode (mass +/-10%)")
        ax.legend(fontsize=8)
        n_trials = len(mc_robustness["mc_bode_mag"])
        ax.text(0.02, 0.02, f"{n_trials} perturbed samples",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No Monte Carlo data", transform=ax.transAxes, ha="center")
    apply_grid(ax)

    # ---- (3,1) Closed-loop pole scatter ----
    ax = axes[3, 1]
    if mc_robustness is not None and len(mc_robustness.get("mc_cl_poles", [])) > 0:
        for i, poles in enumerate(mc_robustness["mc_cl_poles"]):
            ax.plot(poles.real, poles.imag, ".", color="tab:gray", ms=4, alpha=0.4,
                    label="Perturbed" if i == 0 else None)
        nom_poles = mc_robustness["nominal_cl_poles"]
        ax.plot(nom_poles.real, nom_poles.imag, "bo", ms=7, mfc="none", mew=1.5,
                label="Nominal")
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_title("CL Pole Scatter (mass +/-10%)")
        ax.legend(fontsize=8)
        # Check if all perturbed poles are in LHP
        all_stable = all(np.all(p.real < 0) for p in mc_robustness["mc_cl_poles"])
        ax.text(0.98, 0.02,
                f"All in LHP: {'YES' if all_stable else 'NO'}",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                color="green" if all_stable else "red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No Monte Carlo data", transform=ax.transAxes, ha="center")
    apply_grid(ax)

    return fig
