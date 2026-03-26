"""Control-analysis plots with detailed annotations."""

import numpy as np
import matplotlib.pyplot as plt

from visualization.common.axis_style import apply_grid, apply_zero_line


def show_control_plots(t, u_ctrl, u_dist, dt, freq_data):
    """Return a figure with a 4x2 grid of annotated control analysis plots."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), tight_layout=True)
    fig.suptitle("Control Analysis -- LQR Closed-Loop", fontsize=14, fontweight="bold")

    # ---- (0,0) Force comparison ----
    ax = axes[0, 0]
    ax.plot(t, u_ctrl, label="Control", color="tab:blue", lw=0.8)
    ax.plot(t, u_dist, label="Disturbance", color="tab:red", alpha=0.7, lw=0.6)
    ax.set_ylabel("Force [N]")
    ax.set_title("Force Comparison")
    ax.legend(fontsize=8)
    # Annotate peak control force
    i_peak = np.argmax(np.abs(u_ctrl))
    ax.annotate(f"Peak: {u_ctrl[i_peak]:.0f} N",
                xy=(t[i_peak], u_ctrl[i_peak]), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="tab:blue"),
                xytext=(t[i_peak]+0.5, u_ctrl[i_peak]*0.7))
    apply_grid(ax); apply_zero_line(ax)

    # ---- (0,1) Frequency spectrum ----
    ax = axes[0, 1]
    N = len(u_ctrl)
    freqs = np.fft.rfftfreq(N, d=dt)
    fft_ctrl = np.abs(np.fft.rfft(u_ctrl)) * 2 / N
    fft_dist = np.abs(np.fft.rfft(u_dist)) * 2 / N
    ax.semilogy(freqs[1:], fft_ctrl[1:], label="Control", color="tab:blue", lw=0.8)
    ax.semilogy(freqs[1:], fft_dist[1:], label="Disturbance", color="tab:red", alpha=0.7, lw=0.6)
    # Annotate dominant frequency
    i_dom = np.argmax(fft_ctrl[1:]) + 1
    ax.annotate(f"Dominant: {freqs[i_dom]:.1f} Hz",
                xy=(freqs[i_dom], fft_ctrl[i_dom]), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="tab:blue"),
                xytext=(freqs[i_dom]+2, fft_ctrl[i_dom]))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [N]")
    ax.set_title("Frequency Spectrum")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 25)
    apply_grid(ax)

    if freq_data is None:
        for r in range(1, 4):
            for c in range(2):
                axes[r, c].text(0.5, 0.5, "No frequency data",
                                transform=axes[r, c].transAxes, ha="center")
        return fig

    w = freq_data["w"]
    L_jw = freq_data["L_jw"]
    L_mag = freq_data["L_mag"]
    L_phase = freq_data["L_phase_deg"]
    pm = freq_data.get("phase_margin", np.nan)
    gm = freq_data.get("gain_margin_dB", np.nan)
    wgc = freq_data.get("wgc", np.nan)
    wpc = freq_data.get("wpc", np.nan)

    # ---- (1,0) Bode open-loop with GM/PM + slope annotations ----
    ax = axes[1, 0]
    mag_dB = 20 * np.log10(L_mag + 1e-30)
    ax.semilogx(w, mag_dB, "tab:blue", lw=1)
    ax.set_ylabel("Magnitude [dB]", color="tab:blue")
    ax.set_title("Bode -- Open Loop L(jw)")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    apply_grid(ax); apply_zero_line(ax)

    # GM/PM crossover lines and annotations
    info_lines = []
    if not np.isnan(wgc):
        ax.axvline(wgc, color="green", ls=":", lw=1, alpha=0.8)
        info_lines.append(f"PM = {pm:.1f} deg @ wgc = {wgc:.2f} rad/s")
    if not np.isnan(wpc):
        ax.axvline(wpc, color="orange", ls=":", lw=1, alpha=0.8)
        info_lines.append(f"GM = {gm:.1f} dB @ wpc = {wpc:.2f} rad/s")

    # Asymptotic slope annotation: find slope at gain crossover
    if not np.isnan(wgc):
        i_gc = np.argmin(np.abs(w - wgc))
        if 5 < i_gc < len(w) - 5:
            dw = np.log10(w[i_gc+5]) - np.log10(w[i_gc-5])
            dm = mag_dB[i_gc+5] - mag_dB[i_gc-5]
            slope = dm / dw  # dB/decade
            info_lines.append(f"Slope @ wgc: {slope:.0f} dB/dec")

    # Bandwidth (-3dB of CL)
    mag_cl = freq_data.get("mag_cl")
    w_cl = freq_data.get("w_cl")
    if mag_cl is not None and w_cl is not None:
        bw_idx = np.where(mag_cl < -3.0)[0]
        if len(bw_idx) > 0:
            bw = w_cl[bw_idx[0]]
            info_lines.append(f"Bandwidth (-3dB): {bw:.2f} rad/s")

    if info_lines:
        ax.text(0.02, 0.02, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    ax2 = ax.twinx()
    ax2.semilogx(w, L_phase, "tab:orange", alpha=0.7, lw=0.8)
    ax2.axhline(-180, color="tab:orange", ls="--", lw=0.5)
    ax2.set_ylabel("Phase [deg]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # ---- (1,1) Nyquist ----
    ax = axes[1, 1]
    # Plot the full Nyquist contour
    ax.plot(L_jw.real, L_jw.imag, "tab:blue", lw=0.8, label="L(jw), w>0")
    ax.plot(L_jw.real, -L_jw.imag, "tab:blue", ls="--", lw=0.5, alpha=0.5, label="L(jw), w<0")

    # Critical point and unit circle
    theta_c = np.linspace(0, 2*np.pi, 200)
    ax.plot(-1 + np.cos(theta_c), np.sin(theta_c), "r--", lw=0.7, alpha=0.5)
    ax.plot(-1, 0, "rx", ms=10, mew=2, label="(-1,0)")

    # Annotate closest approach to -1
    dist_to_m1 = np.abs(L_jw - (-1 + 0j))
    i_closest = np.argmin(dist_to_m1)
    min_dist = dist_to_m1[i_closest]
    ax.annotate(f"Min dist to (-1,0): {min_dist:.3f}\nw = {w[i_closest]:.2f} rad/s",
                xy=(L_jw[i_closest].real, L_jw[i_closest].imag),
                xytext=(0.05, 0.05), textcoords="axes fraction",
                fontsize=7, arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # Direction arrows
    n_arr = len(L_jw)
    for frac in [0.1, 0.3, 0.6]:
        idx = int(frac * n_arr)
        if idx + 1 < n_arr:
            ax.annotate("", xy=(L_jw[idx+1].real, L_jw[idx+1].imag),
                        xytext=(L_jw[idx].real, L_jw[idx].imag),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5))

    ax.set_xlabel("Real"); ax.set_ylabel("Imag")
    ax.set_title("Nyquist Diagram")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    # Auto-scale with padding
    r_max = max(3.0, np.percentile(np.abs(L_jw), 95) * 1.2)
    ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
    apply_grid(ax)

    # ---- (2,0) Sensitivity S(jw) & T(jw) ----
    ax = axes[2, 0]
    S_mag = np.abs(freq_data["S_jw"])
    T_mag = np.abs(freq_data["T_jw"])
    S_dB = 20 * np.log10(S_mag + 1e-30)
    T_dB = 20 * np.log10(T_mag + 1e-30)
    ax.semilogx(w, S_dB, label="S(jw)", color="tab:blue", lw=1)
    ax.semilogx(w, T_dB, label="T(jw)", color="tab:red", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(6, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.text(w[0]*2, 6.5, "6 dB limit", fontsize=6, color="gray")

    Ms = freq_data.get("Ms", 1.0)
    Mt = freq_data.get("Mt", 1.0)
    # Peak markers
    i_Ms = np.argmax(S_mag)
    i_Mt = np.argmax(T_mag)
    ax.plot(w[i_Ms], S_dB[i_Ms], "v", color="tab:blue", ms=6)
    ax.plot(w[i_Mt], T_dB[i_Mt], "v", color="tab:red", ms=6)

    info = [f"Ms = {20*np.log10(Ms):.1f} dB @ {w[i_Ms]:.1f} rad/s",
            f"Mt = {20*np.log10(Mt):.1f} dB @ {w[i_Mt]:.1f} rad/s"]
    # S-T crossover frequency
    cross_idx = np.where(np.diff(np.sign(S_dB - T_dB)))[0]
    if len(cross_idx) > 0:
        w_cross = w[cross_idx[0]]
        info.append(f"S-T crossover: {w_cross:.2f} rad/s")

    ax.text(0.02, 0.02, "\n".join(info), transform=ax.transAxes,
            fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Sensitivity & Complementary Sensitivity")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # ---- (2,1) Pole map ----
    ax = axes[2, 1]
    poles_ol = freq_data["poles_ol"]
    poles_cl = freq_data["poles_cl"]
    ax.plot(poles_ol.real, poles_ol.imag, "rx", ms=10, mew=2, label="OL poles")
    ax.plot(poles_cl.real, poles_cl.imag, "bo", ms=7, mfc="none", mew=1.5, label="CL poles")
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)

    # Annotate each CL pole with damping ratio
    for p in poles_cl:
        wn = np.abs(p)
        if wn > 1e-6:
            zeta = -p.real / wn
            ax.annotate(f"z={zeta:.2f}", xy=(p.real, p.imag),
                        xytext=(5, 5), textcoords="offset points", fontsize=6, color="blue")

    # Dominant pole (closest to imaginary axis)
    i_dom = np.argmax(poles_cl.real)
    ax.annotate(f"Dominant: {poles_cl[i_dom]:.2f}\nRe={poles_cl[i_dom].real:.2f}",
                xy=(poles_cl[i_dom].real, poles_cl[i_dom].imag),
                xytext=(0.65, 0.85), textcoords="axes fraction",
                fontsize=7, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    ax.set_xlabel("Real"); ax.set_ylabel("Imag")
    ax.set_title("Pole Map (damping ratio annotated)")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # ---- (3,0) Bode closed-loop ----
    ax = axes[3, 0]
    w_cl = freq_data["w_cl"]
    mag_cl = freq_data["mag_cl"]
    phase_cl = freq_data["phase_cl"]
    ax.semilogx(w_cl, mag_cl, "tab:blue", lw=1)
    ax.set_ylabel("Magnitude [dB]", color="tab:blue")
    ax.set_title("Bode -- Closed Loop (Dist -> Cart)")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    # -3dB bandwidth
    bw_idx = np.where(mag_cl < mag_cl[0] - 3.0)[0]
    if len(bw_idx) > 0:
        bw = w_cl[bw_idx[0]]
        ax.axvline(bw, color="green", ls=":", lw=1, alpha=0.7)
        ax.annotate(f"BW = {bw:.2f} rad/s", xy=(bw, mag_cl[0]-3),
                    xytext=(bw*2, mag_cl[0]-10), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="green"),
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9))

    # Resonance peak
    i_peak = np.argmax(mag_cl)
    if mag_cl[i_peak] > mag_cl[0] + 1:
        ax.plot(w_cl[i_peak], mag_cl[i_peak], "rv", ms=6)
        ax.annotate(f"Peak: {mag_cl[i_peak]:.1f} dB\n@ {w_cl[i_peak]:.1f} rad/s",
                    xy=(w_cl[i_peak], mag_cl[i_peak]),
                    xytext=(w_cl[i_peak]*3, mag_cl[i_peak]-5), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="red"))

    apply_grid(ax); apply_zero_line(ax)
    ax2 = ax.twinx()
    ax2.semilogx(w_cl, phase_cl, "tab:orange", alpha=0.7, lw=0.8)
    ax2.set_ylabel("Phase [deg]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # ---- (3,1) Step response ----
    ax = axes[3, 1]
    t_step = freq_data["t_step"]
    y_step = freq_data["y_step"]
    y_ss = freq_data.get("y_ss", 0)
    overshoot = freq_data.get("overshoot", 0)
    t_settle = freq_data.get("t_settle", 0)

    ax.plot(t_step, y_step, "tab:blue", lw=1)
    ax.axhline(y_ss, color="gray", ls=":", lw=0.7, label=f"SS = {y_ss:.4f} m")

    # 2% settling band
    if abs(y_ss) > 1e-8:
        band = 0.02 * abs(y_ss)
        ax.axhspan(y_ss - band, y_ss + band, alpha=0.1, color="green", label="2% band")

    # Peak annotation
    i_pk = np.argmax(np.abs(y_step))
    ax.plot(t_step[i_pk], y_step[i_pk], "rv", ms=6)
    ax.annotate(f"Peak: {y_step[i_pk]:.4f} m\n@ t={t_step[i_pk]:.2f}s",
                xy=(t_step[i_pk], y_step[i_pk]),
                xytext=(t_step[i_pk]+0.3, y_step[i_pk]*0.7), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="red"))

    # Rise time (10% to 90% of steady-state)
    if abs(y_ss) > 1e-8:
        y10 = 0.1 * y_ss; y90 = 0.9 * y_ss
        i10 = np.where(np.abs(y_step) >= np.abs(y10))[0]
        i90 = np.where(np.abs(y_step) >= np.abs(y90))[0]
        if len(i10) > 0 and len(i90) > 0:
            t_rise = t_step[i90[0]] - t_step[i10[0]]
        else:
            t_rise = np.nan
    else:
        t_rise = np.nan

    info = f"OS = {overshoot:.1f}%\nTs(2%) = {t_settle:.3f} s\nTr = {t_rise:.3f} s"
    ax.text(0.98, 0.98, info, transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    ax.set_xlabel("Time [s]"); ax.set_ylabel("Cart x [m]")
    ax.set_title("Step Response (with time-domain specs)")
    ax.legend(fontsize=7, loc="lower right")
    apply_grid(ax); apply_zero_line(ax)

    return fig
