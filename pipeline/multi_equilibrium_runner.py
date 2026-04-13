"""8개 평형점 전체에 대한 시각화 자동 생성 오케스트레이터.

각 평형점(DDD, DDU, ..., UUU)에 대해:
  - images/{NAME}/ 서브디렉터리 생성
  - animation.gif, dynamics_analysis.png, control_analysis.png,
    lqr_verification.png, roa_analysis.png, comparison_analysis.png 저장
최종적으로 images/summary_grid.png (4x2 비교 그리드) 를 생성한다.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

from parameters.config import SystemConfig
from parameters.equilibrium import EQUILIBRIUM_CONFIGS
from pipeline.save_outputs import save_figure, save_animation, IMAGES_DIR
from pipeline.defaults import (T_END, DT, IMPULSE, DIST_AMPLITUDE,
                               DIST_BANDWIDTH, SEED, U_MAX)
from utils.logger import get_logger
from visualization.common.axis_style import apply_publication_style
from visualization.common.korean_font import apply_korean_font

log = get_logger()

# 평형점 한글 이름 설명
EQUILIBRIUM_KO = {
    "DDD": "하-하-하 (모두 하향)",
    "DDU": "하-하-상",
    "DUD": "하-상-하",
    "DUU": "하-상-상",
    "UDD": "상-하-하",
    "UDU": "상-하-상",
    "UUD": "상-상-하",
    "UUU": "상-상-상 (모두 상향)",
}

# 평형점 안정성 분류 (선형화 기준)
EQUILIBRIUM_STABILITY = {
    "DDD": "자연 안정",
    "DDU": "LQR 필요",
    "DUD": "LQR 필요",
    "DUU": "LQR 필요",
    "UDD": "LQR 필요",
    "UDU": "LQR 필요",
    "UUD": "LQR 필요",
    "UUU": "LQR 필요",
}


def _build_cfg(base_cfg: SystemConfig, eq_name: str) -> SystemConfig:
    """기존 SystemConfig 를 복사하여 target_equilibrium 만 변경한 새 인스턴스를 반환한다."""
    cfg = SystemConfig(
        mc=base_cfg.mc, m1=base_cfg.m1, m2=base_cfg.m2, m3=base_cfg.m3,
        L1=base_cfg.L1, L2=base_cfg.L2, L3=base_cfg.L3, g=base_cfg.g,
        target_equilibrium=eq_name,
        actuator_saturation=base_cfg.actuator_saturation,
    )
    return cfg


def _run_single_equilibrium(
    base_cfg: SystemConfig,
    eq_name: str,
    t_end: float = T_END,
    dt: float = DT,
    impulse: float = IMPULSE,
    dist_amplitude: float = DIST_AMPLITUDE,
    dist_bandwidth: float = DIST_BANDWIDTH,
    seed: int = SEED,
    u_max: float = U_MAX,
) -> dict:
    """단일 평형점에 대해 전체 파이프라인을 실행하고 결과를 반환한다.

    Parameters
    ----------
    base_cfg : SystemConfig
        기본 시스템 파라미터.
    eq_name : str
        대상 평형점 이름 (예: "UUU").

    Returns
    -------
    dict
        저장된 파일 경로 목록, 성공 여부, 소요 시간 등 포함.
    """
    apply_publication_style()
    t_start = time.time()
    log.info("=" * 60)
    log.info("평형점 %s (%s) 처리 시작", eq_name, EQUILIBRIUM_KO[eq_name])
    log.info("=" * 60)

    cfg = _build_cfg(base_cfg, eq_name)
    saved_paths = {}

    try:
        from control.lqr import compute_lqr_gains
        from control.closed_loop import compute_closed_loop
        from control.comparison import compare_controllers
        from simulation.disturbance.generate_disturbance import generate_disturbance
        from simulation.loop.time_loop import simulate
        from analysis.state.derived_state import compute_derived_state
        from analysis.energy.total_energy import compute_energy
        from analysis.frequency.frequency_response import compute_frequency_response
        from analysis.lqr_verification.compute_verification import (
            compute_lqr_verification, compute_monte_carlo_robustness)
        from analysis.region_of_attraction import estimate_roa
        from analysis.gain_scheduling_stability import verify_gain_scheduling_stability
        from control.gain_scheduling import GainScheduler

        # 1. LQR 게인 계산
        log.info("  LQR 게인 계산 중...")
        K, A, B, P, Q, R = compute_lqr_gains(cfg)
        cl = compute_closed_loop(A, B, K)
        log.info("  폐루프 극점 안정: %s", cl['is_stable'])

        # 2. 게인 스케줄러
        gs = GainScheduler(cfg)

        # 3. 제어기 비교
        comparison_data = compare_controllers(A, B, K)

        # 4. 외란 생성
        t_tmp = np.linspace(0, t_end, int(t_end / dt) + 1)
        dist = generate_disturbance(t_tmp, amplitude=dist_amplitude,
                                    bandwidth=dist_bandwidth, seed=seed)

        # 5. 시뮬레이션
        log.info("  시뮬레이션 실행 중 (%.1f s, dt=%.4f)...", t_end, dt)
        t, q, dq, u_ctrl, u_dist, u_raw_peak, n_sat = simulate(
            cfg, K, t_end=t_end, dt=dt,
            impulse=impulse, disturbance=dist,
            gain_scheduler=gs, u_max=u_max)

        # 6. 분석
        state = compute_derived_state(cfg, q, dq)
        energy = compute_energy(cfg, q, dq,
                                state["phi1"], state["phi2"], state["phi3"])
        A_cl = cl["A_cl"]
        freq_data = compute_frequency_response(A, B, K, A_cl)
        lqr_verif = compute_lqr_verification(t, q, dq, cfg.equilibrium, K, A, B, P, Q, R)
        mc_robustness = compute_monte_carlo_robustness(cfg)

        # 7. ROA 및 게인 스케줄링 안정성
        log.info("  ROA 추정 중 (n=300)...")
        roa_data = estimate_roa(cfg, K, n_samples=300, max_angle_deg=45, t_horizon=5.0)
        gs_stability = verify_gain_scheduling_stability(cfg, gs)

        # 8. 시각화 생성 및 저장
        from visualization.animation.show_animation import show_animation
        from visualization.dynamics_plots.show_dynamics_plots import show_dynamics_plots
        from visualization.control_plots.show_control_plots import show_control_plots
        from visualization.lqr_plots.show_lqr_plots import show_lqr_plots
        from visualization.roa_plots.show_roa_plots import show_roa_plots
        from visualization.comparison_plots.show_comparison_plots import show_comparison_plots

        # 애니메이션
        log.info("  애니메이션 생성 중...")
        fig_anim, ani = _make_annotated_animation(cfg, t, state, dt, eq_name)
        path = save_animation(ani, "animation", fps=30, subdir=eq_name)
        saved_paths["animation"] = path
        plt.close(fig_anim)

        # 동역학 분석 (한글 제목)
        log.info("  dynamics_analysis 생성 중...")
        fig_dyn = _make_dynamics_plots_ko(t, q, dq, state, energy, u_ctrl, eq_name)
        path = save_figure(fig_dyn, "dynamics_analysis", subdir=eq_name)
        saved_paths["dynamics_analysis"] = path
        plt.close(fig_dyn)

        # 제어 분석 (한글 제목)
        log.info("  control_analysis 생성 중...")
        fig_ctrl = _make_control_plots_ko(t, u_ctrl, u_dist, dt, freq_data, eq_name)
        path = save_figure(fig_ctrl, "control_analysis", subdir=eq_name)
        saved_paths["control_analysis"] = path
        plt.close(fig_ctrl)

        # LQR 검증 (한글 제목)
        log.info("  lqr_verification 생성 중...")
        fig_lqr = _make_lqr_plots_ko(lqr_verif, mc_robustness, eq_name)
        path = save_figure(fig_lqr, "lqr_verification", subdir=eq_name)
        saved_paths["lqr_verification"] = path
        plt.close(fig_lqr)

        # ROA 분석 (한글 제목)
        log.info("  roa_analysis 생성 중...")
        fig_roa = _make_roa_plots_ko(roa_data, gs_stability, eq_name)
        path = save_figure(fig_roa, "roa_analysis", subdir=eq_name)
        saved_paths["roa_analysis"] = path
        plt.close(fig_roa)

        # 비교 분석 (한글 제목)
        log.info("  comparison_analysis 생성 중...")
        fig_cmp = _make_comparison_plots_ko(comparison_data, eq_name)
        path = save_figure(fig_cmp, "comparison_analysis", subdir=eq_name)
        saved_paths["comparison_analysis"] = path
        plt.close(fig_cmp)

        elapsed = time.time() - t_start
        log.info("  평형점 %s 완료: %.1f 초", eq_name, elapsed)

        return {
            "eq_name": eq_name,
            "success": True,
            "elapsed": elapsed,
            "saved_paths": saved_paths,
            "roa_rate": roa_data["success_rate"],
            "max_dev_deg": roa_data["max_stable_deviation_deg"],
            "cl_stable": bool(cl["is_stable"]),
            "settling_time": _compute_settling_time(t, q[:, 0]),
        }

    except Exception as exc:
        elapsed = time.time() - t_start
        log.error("  평형점 %s 실패: %s", eq_name, exc, exc_info=True)
        return {
            "eq_name": eq_name,
            "success": False,
            "elapsed": elapsed,
            "saved_paths": saved_paths,
            "error": str(exc),
        }


def _compute_settling_time(t: np.ndarray, x: np.ndarray,
                            threshold: float = 0.02) -> float:
    """카트 위치 x 가 마지막 값의 threshold 배 이내로 정착하는 시간을 반환한다."""
    x_ss = x[-1]
    band = max(abs(x_ss) * threshold, 0.005)  # 최소 5mm
    outside = np.where(np.abs(x - x_ss) > band)[0]
    if len(outside) == 0:
        return 0.0
    return float(t[outside[-1]])


# ────────────────────────────────────────────────
#  한글 제목 시각화 래퍼 함수들
# ────────────────────────────────────────────────

def _make_annotated_animation(cfg, t, state, dt, eq_name):
    """평형점 정보 주석이 포함된 애니메이션을 생성한다."""
    from visualization.animation.show_animation import show_animation
    fig, ani = show_animation(cfg, t, state, dt=dt)

    ax = fig.axes[0]
    ko_name = EQUILIBRIUM_KO[eq_name]
    ax.set_title(f"카트 + 삼중 역진자 애니메이션 — {eq_name} 평형점")
    ax.text(0.02, 0.02,
            f"평형점: {eq_name}\n설명: {ko_name}\n초기 편차: 5°",
            transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="gray", alpha=0.9))
    return fig, ani


def _make_dynamics_plots_ko(t, q, dq, state, energy, u_ctrl, eq_name):
    """한글 제목이 적용된 동역학 분석 figure를 생성한다.

    5x2 = 10개 서브플롯 구성:
    (0,0) 카트 위치  (0,1) 카트 속도
    (1,0) 각도 편차  (1,1) 각속도
    (2,0) 카트 가속도 (2,1) 각가속도
    (3,0) 에너지 (KE/PE/TE)  (3,1) 위상도
    (4,0) 제어력  (4,1) 일반화 가속도
    """
    from visualization.common.colors import LINK_COLORS
    from visualization.common.axis_style import apply_grid, apply_zero_line

    ko = EQUILIBRIUM_KO[eq_name]
    fig, axes = plt.subplots(5, 2, figsize=(15, 18), tight_layout=True)
    fig.suptitle(
        f"동역학 분석 — {eq_name} 평형점 ({ko})",
        fontsize=14, fontweight="bold"
    )

    link_names = ["링크1", "링크2", "링크3"]
    dt_val = t[1] - t[0]
    ddq = np.gradient(dq, dt_val, axis=0)

    # (0,0) 카트 위치
    ax = axes[0, 0]
    ax.plot(t, q[:, 0], "k", lw=1.5)
    ax.set_ylabel("위치 [m]")
    ax.set_title("카트 위치")
    apply_grid(ax); apply_zero_line(ax)

    # (0,1) 카트 속도
    ax = axes[0, 1]
    ax.plot(t, dq[:, 0], "k", lw=1.5)
    ax.set_ylabel("속도 [m/s]")
    ax.set_title("카트 속도")
    apply_grid(ax); apply_zero_line(ax)

    # (1,0) 각도 편차
    ax = axes[1, 0]
    for i, key in enumerate(["dth1", "dth2", "dth3"]):
        ax.plot(t, np.degrees(state[key]), color=LINK_COLORS[i],
                lw=1.5, label=link_names[i])
    ax.set_ylabel("편차 [deg]")
    ax.set_title("링크 각도 편차 (평형점 기준)")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (1,1) 각속도
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(t, np.degrees(dq[:, i + 1]), color=LINK_COLORS[i],
                lw=1.5, label=link_names[i])
    ax.set_ylabel("각속도 [deg/s]")
    ax.set_title("링크 각속도")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (2,0) 카트 가속도
    ax = axes[2, 0]
    ax.plot(t, ddq[:, 0], "k", lw=0.8)
    ax.set_ylabel("가속도 [m/s²]")
    ax.set_title("카트 가속도")
    apply_grid(ax); apply_zero_line(ax)

    # (2,1) 각가속도
    ax = axes[2, 1]
    for i, (col, nm) in enumerate(zip(LINK_COLORS, link_names)):
        ax.plot(t, np.degrees(ddq[:, i + 1]), color=col, lw=0.8, label=nm)
    ax.set_ylabel("각가속도 [deg/s²]")
    ax.set_title("링크 각가속도")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (3,0) 에너지 (KE/PE/TE)
    ax = axes[3, 0]
    ax.plot(t, energy["KE"], label="운동에너지 KE", color="tab:orange", lw=1.5)
    ax.plot(t, energy["PE"], label="위치에너지 PE", color="tab:cyan", lw=1.5)
    ax.plot(t, energy["TE"], label="전체에너지 TE", color="k", ls="--", lw=1.5)
    ax.set_xlabel("시간 [s]")
    ax.set_ylabel("에너지 [J]")
    ax.set_title("시스템 에너지 (KE / PE / TE)")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (3,1) 위상도
    ax = axes[3, 1]
    for i, (key, nm) in enumerate(
            zip(["dth1", "dth2", "dth3"], link_names)):
        ax.plot(np.degrees(state[key]), np.degrees(dq[:, i + 1]),
                color=LINK_COLORS[i], lw=0.5, alpha=0.8, label=nm)
    ax.plot(0, 0, "k+", ms=12, mew=2)
    ax.set_xlabel("각도 오차 [deg]")
    ax.set_ylabel("각속도 [deg/s]")
    ax.set_title("위상도 (각도 기준)")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (4,0) 제어력
    ax = axes[4, 0]
    if u_ctrl is not None:
        ax.plot(t, u_ctrl, "tab:blue", lw=1.5)
        i_peak = np.argmax(np.abs(u_ctrl))
        ax.annotate(f"최대: {u_ctrl[i_peak]:.1f} N",
                    xy=(t[i_peak], u_ctrl[i_peak]), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="tab:blue"),
                    xytext=(t[i_peak] + 0.3, u_ctrl[i_peak] * 0.7))
    else:
        ax.text(0.5, 0.5, "제어 데이터 없음",
                transform=ax.transAxes, ha="center")
    ax.set_xlabel("시간 [s]")
    ax.set_ylabel("힘 [N]")
    ax.set_title("제어력")
    apply_grid(ax); apply_zero_line(ax)

    # (4,1) 일반화 가속도 (관절 반력 추정)
    ax = axes[4, 1]
    ax.plot(t, ddq[:, 0], "k", lw=0.8, label="카트 (ddx)")
    ax.plot(t, ddq[:, 1], color=LINK_COLORS[0], lw=0.8, label="관절1 (ddth1)")
    ax.plot(t, ddq[:, 2], color=LINK_COLORS[1], lw=0.8, label="관절2 (ddth2)")
    ax.plot(t, ddq[:, 3], color=LINK_COLORS[2], lw=0.8, label="관절3 (ddth3)")
    ax.set_xlabel("시간 [s]")
    ax.set_ylabel("일반화 가속도 [rad/s²]")
    ax.set_title("관절 반력 추정 (일반화 가속도)")
    ax.legend(fontsize=7)
    apply_grid(ax); apply_zero_line(ax)

    return fig


def _make_control_plots_ko(t, u_ctrl, u_dist, dt, freq_data, eq_name):
    """한글 제목이 적용된 제어 분석 figure를 생성한다."""
    from visualization.common.axis_style import apply_grid, apply_zero_line

    ko = EQUILIBRIUM_KO[eq_name]
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), tight_layout=True)
    fig.suptitle(
        f"LQR 폐루프 제어 분석 — {eq_name} 평형점 ({ko})",
        fontsize=14, fontweight="bold"
    )

    # (0,0) 힘 비교
    ax = axes[0, 0]
    ax.plot(t, u_ctrl, label="제어력", color="tab:blue", lw=0.8)
    ax.plot(t, u_dist, label="외란", color="tab:red", alpha=0.7, lw=0.6)
    ax.set_ylabel("힘 [N]")
    ax.set_title("제어력 vs 외란 비교")
    ax.legend(fontsize=8)
    if len(u_ctrl) > 0:
        i_peak = np.argmax(np.abs(u_ctrl))
        ax.annotate(f"최대: {u_ctrl[i_peak]:.0f} N",
                    xy=(t[i_peak], u_ctrl[i_peak]), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="tab:blue"),
                    xytext=(t[i_peak] + 0.5, u_ctrl[i_peak] * 0.7))
    apply_grid(ax); apply_zero_line(ax)

    # (0,1) 주파수 스펙트럼
    ax = axes[0, 1]
    N = len(u_ctrl)
    freqs = np.fft.rfftfreq(N, d=dt)
    fft_ctrl = np.abs(np.fft.rfft(u_ctrl)) * 2 / N
    fft_dist = np.abs(np.fft.rfft(u_dist)) * 2 / N
    ax.semilogy(freqs[1:], fft_ctrl[1:], label="제어력", color="tab:blue", lw=0.8)
    ax.semilogy(freqs[1:], fft_dist[1:], label="외란", color="tab:red", alpha=0.7, lw=0.6)
    if len(fft_ctrl) > 1:
        i_dom = np.argmax(fft_ctrl[1:]) + 1
        ax.annotate(f"지배 주파수: {freqs[i_dom]:.1f} Hz",
                    xy=(freqs[i_dom], fft_ctrl[i_dom]), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="tab:blue"),
                    xytext=(freqs[i_dom] + 2, fft_ctrl[i_dom]))
    ax.set_xlabel("주파수 [Hz]")
    ax.set_ylabel("진폭 [N]")
    ax.set_title("주파수 스펙트럼")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 25)
    apply_grid(ax)

    if freq_data is None:
        for r in range(1, 4):
            for c in range(2):
                axes[r, c].text(0.5, 0.5, "주파수 데이터 없음",
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

    # (1,0) Bode 선도 (개루프)
    ax = axes[1, 0]
    mag_dB = 20 * np.log10(L_mag + 1e-30)
    ax.semilogx(w, mag_dB, "tab:blue", lw=1)
    ax.set_ylabel("진폭 [dB]", color="tab:blue")
    ax.set_title("Bode 선도 — 개루프 L(jω)")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    apply_grid(ax); apply_zero_line(ax)

    info_lines = []
    if not np.isnan(wgc):
        ax.axvline(wgc, color="green", ls=":", lw=1, alpha=0.8)
        info_lines.append(f"위상여유 PM = {pm:.1f}° @ ωgc = {wgc:.2f} rad/s")
    if not np.isnan(wpc):
        ax.axvline(wpc, color="orange", ls=":", lw=1, alpha=0.8)
        info_lines.append(f"이득여유 GM = {gm:.1f} dB @ ωpc = {wpc:.2f} rad/s")
    if info_lines:
        ax.text(0.02, 0.02, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    ax2 = ax.twinx()
    ax2.semilogx(w, L_phase, "tab:orange", alpha=0.7, lw=0.8)
    ax2.axhline(-180, color="tab:orange", ls="--", lw=0.5)
    ax2.set_ylabel("위상 [deg]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # (1,1) Nyquist 선도
    ax = axes[1, 1]
    ax.plot(L_jw.real, L_jw.imag, "tab:blue", lw=0.8, label="L(jω), ω>0")
    ax.plot(L_jw.real, -L_jw.imag, "tab:blue", ls="--", lw=0.5, alpha=0.5, label="L(jω), ω<0")
    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(-1 + np.cos(theta_c), np.sin(theta_c), "r--", lw=0.7, alpha=0.5)
    ax.plot(-1, 0, "rx", ms=10, mew=2, label="(-1, 0)")
    dist_to_m1 = np.abs(L_jw - (-1 + 0j))
    i_closest = np.argmin(dist_to_m1)
    ax.annotate(
        f"(-1,0)까지 최소 거리: {dist_to_m1[i_closest]:.3f}\nω = {w[i_closest]:.2f} rad/s",
        xy=(L_jw[i_closest].real, L_jw[i_closest].imag),
        xytext=(0.05, 0.05), textcoords="axes fraction",
        fontsize=7, arrowprops=dict(arrowstyle="->", color="red"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax.set_xlabel("실수부"); ax.set_ylabel("허수부")
    ax.set_title("Nyquist 선도")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    r_max = max(3.0, np.percentile(np.abs(L_jw), 95) * 1.2)
    ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
    apply_grid(ax)

    # (2,0) 감도 함수 S(jω) & T(jω)
    ax = axes[2, 0]
    S_mag = np.abs(freq_data["S_jw"])
    T_mag = np.abs(freq_data["T_jw"])
    S_dB = 20 * np.log10(S_mag + 1e-30)
    T_dB = 20 * np.log10(T_mag + 1e-30)
    ax.semilogx(w, S_dB, label="S(jω) 감도", color="tab:blue", lw=1)
    ax.semilogx(w, T_dB, label="T(jω) 상보 감도", color="tab:red", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(6, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.text(w[0] * 2, 6.5, "6 dB 한계", fontsize=6, color="gray")
    Ms = freq_data.get("Ms", 1.0)
    Mt = freq_data.get("Mt", 1.0)
    i_Ms = np.argmax(S_mag)
    i_Mt = np.argmax(T_mag)
    ax.plot(w[i_Ms], S_dB[i_Ms], "v", color="tab:blue", ms=6)
    ax.plot(w[i_Mt], T_dB[i_Mt], "v", color="tab:red", ms=6)
    info = [f"Ms = {20*np.log10(Ms):.1f} dB @ {w[i_Ms]:.1f} rad/s",
            f"Mt = {20*np.log10(Mt):.1f} dB @ {w[i_Mt]:.1f} rad/s"]
    ax.text(0.02, 0.02, "\n".join(info), transform=ax.transAxes,
            fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax.set_xlabel("주파수 [rad/s]")
    ax.set_ylabel("진폭 [dB]")
    ax.set_title("감도 & 상보 감도 함수")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (2,1) 극점 지도
    ax = axes[2, 1]
    poles_ol = freq_data["poles_ol"]
    poles_cl = freq_data["poles_cl"]
    ax.plot(poles_ol.real, poles_ol.imag, "rx", ms=10, mew=2, label="개루프 극점")
    ax.plot(poles_cl.real, poles_cl.imag, "bo", ms=7, mfc="none", mew=1.5, label="폐루프 극점")
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    for p in poles_cl:
        wn = np.abs(p)
        if wn > 1e-6:
            zeta = -p.real / wn
            ax.annotate(f"ζ={zeta:.2f}", xy=(p.real, p.imag),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6, color="blue")
    ax.set_xlabel("실수부"); ax.set_ylabel("허수부")
    ax.set_title("극점 지도 (감쇠비 주석)")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (3,0) 폐루프 Bode 선도
    ax = axes[3, 0]
    w_cl = freq_data["w_cl"]
    mag_cl = freq_data["mag_cl"]
    phase_cl = freq_data["phase_cl"]
    ax.semilogx(w_cl, mag_cl, "tab:blue", lw=1)
    ax.set_ylabel("진폭 [dB]", color="tab:blue")
    ax.set_title("Bode 선도 — 폐루프 (외란 → 카트)")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    bw_idx = np.where(mag_cl < mag_cl[0] - 3.0)[0]
    if len(bw_idx) > 0:
        bw = w_cl[bw_idx[0]]
        ax.axvline(bw, color="green", ls=":", lw=1, alpha=0.7)
        ax.annotate(f"대역폭 = {bw:.2f} rad/s", xy=(bw, mag_cl[0] - 3),
                    xytext=(bw * 2, mag_cl[0] - 10), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="green"),
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9))
    apply_grid(ax); apply_zero_line(ax)
    ax2 = ax.twinx()
    ax2.semilogx(w_cl, phase_cl, "tab:orange", alpha=0.7, lw=0.8)
    ax2.set_ylabel("위상 [deg]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # (3,1) 계단 응답
    ax = axes[3, 1]
    t_step = freq_data["t_step"]
    y_step = freq_data["y_step"]
    y_ss = freq_data.get("y_ss", 0)
    overshoot = freq_data.get("overshoot", 0)
    t_settle = freq_data.get("t_settle", 0)
    ax.plot(t_step, y_step, "tab:blue", lw=1)
    ax.axhline(y_ss, color="gray", ls=":", lw=0.7, label=f"정상상태 = {y_ss:.4f} m")
    if abs(y_ss) > 1e-8:
        band = 0.02 * abs(y_ss)
        ax.axhspan(y_ss - band, y_ss + band, alpha=0.1, color="green", label="±2% 범위")
    info = f"초과량 = {overshoot:.1f}%\n정착시간(2%) = {t_settle:.3f} s"
    ax.text(0.98, 0.98, info, transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax.set_xlabel("시간 [s]"); ax.set_ylabel("카트 위치 [m]")
    ax.set_title("계단 응답 (시간 영역 성능 지표)")
    ax.legend(fontsize=7, loc="lower right")
    apply_grid(ax); apply_zero_line(ax)

    return fig


def _make_lqr_plots_ko(lqr_verif, mc_robustness, eq_name):
    """한글 제목이 적용된 LQR 검증 figure를 생성한다."""
    from visualization.common.axis_style import apply_grid, apply_zero_line

    ko = EQUILIBRIUM_KO[eq_name]
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), tight_layout=True)
    fig.suptitle(
        f"LQR 검증 — {eq_name} 평형점 ({ko})",
        fontsize=14, fontweight="bold"
    )

    t = lqr_verif["t"]

    # (0,0) 리아푸노프 함수 V(t)
    ax = axes[0, 0]
    V = lqr_verif["lyapunov_V"]
    ax.semilogy(t, V, "tab:blue", lw=1)
    ax.set_ylabel("V(t) = z'Pz")
    ax.set_title("리아푸노프 함수 (단조 감소 필요)")
    apply_grid(ax)
    dV = np.diff(V)
    n_increase = np.sum(dV > 1e-10)
    pct_decrease = (1 - n_increase / len(dV)) * 100
    color = "green" if pct_decrease > 95 else "red"
    ax.text(0.98, 0.98, f"감소 비율: {pct_decrease:.1f}%",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (0,1) Riccati P 고유값
    ax = axes[0, 1]
    P_eig = lqr_verif["P_eigenvalues"]
    idx = np.arange(len(P_eig))
    colors = ["green" if v > 0 else "red" for v in P_eig]
    ax.bar(idx, P_eig, color=colors, alpha=0.7)
    ax.set_xlabel("고유값 인덱스")
    ax.set_ylabel("값")
    ax.set_title("Riccati P 행렬 고유값 (모두 양수 필요)")
    ax.set_xticks(idx)
    apply_grid(ax)
    all_pos = np.all(P_eig > 0)
    ax.text(0.98, 0.98,
            f"P 양정치: {'예(YES)' if all_pos else '아니오(NO)'}",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            color="green" if all_pos else "red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (1,0) LQR 비용 분해
    ax = axes[1, 0]
    ax.semilogy(t, lqr_verif["cost_state"], label="상태 비용 (z'Qz)",
                color="tab:blue", lw=0.8)
    ax.semilogy(t, lqr_verif["cost_control"], label="제어 비용 (u'Ru)",
                color="tab:red", lw=0.8)
    ax.set_ylabel("순간 비용")
    ax.set_title("LQR 비용 분해")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (1,1) 누적 비용 J(t)
    ax = axes[1, 1]
    J = lqr_verif["cost_cumulative"]
    ax.plot(t, J, "tab:blue", lw=1)
    ax.set_ylabel("J(t) = ∫(z'Qz + u'Ru)dt")
    ax.set_title("누적 LQR 비용 (수렴 확인)")
    apply_grid(ax)
    ax.axhline(J[-1], color="gray", ls=":", lw=0.7)
    ax.text(0.98, 0.5, f"최종값 J = {J[-1]:.2f}",
            transform=ax.transAxes, fontsize=9, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (2,0) 복귀 차분 |1 + L(jω)| — Kalman 부등식
    ax = axes[2, 0]
    w_rd = lqr_verif["return_diff_w"]
    rd = lqr_verif["return_difference"]
    ax.semilogx(w_rd, 20 * np.log10(rd), "tab:blue", lw=1)
    ax.axhline(0, color="red", ls="--", lw=1, label="|1+L| = 1 (0 dB)")
    ax.set_xlabel("주파수 [rad/s]")
    ax.set_ylabel("|1 + L(jω)| [dB]")
    ax.set_title("복귀 차분 (Kalman 부등식: 0 dB 이상 필요)")
    ax.legend(fontsize=8)
    apply_grid(ax)
    min_rd = np.min(rd)
    min_rd_dB = 20 * np.log10(min_rd)
    i_min = np.argmin(rd)
    kalman_ok = min_rd >= 1.0 - 1e-6
    ax.text(0.98, 0.02,
            f"최소 |1+L| = {min_rd_dB:.2f} dB @ {w_rd[i_min]:.1f} rad/s\n"
            f"Kalman 부등식: {'만족(OK)' if kalman_ok else '위반(FAIL)'}",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="green" if kalman_ok else "red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (2,1) Nyquist 검증
    ax = axes[2, 1]
    L_jw = lqr_verif["L_jw_full"]
    ax.plot(L_jw.real, L_jw.imag, "tab:blue", lw=0.8, label="L(jω), ω>0")
    ax.plot(L_jw.real, -L_jw.imag, "tab:blue", ls="--", lw=0.5, alpha=0.5, label="L(jω), ω<0")
    ax.plot(-1, 0, "rx", ms=10, mew=2)
    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(-1 + np.cos(theta_c), np.sin(theta_c), "r--", lw=0.5, alpha=0.3)
    n_enc = lqr_verif["n_encirclements_cw"]
    n_unstable = lqr_verif["n_unstable_ol"]
    nyq_ok = lqr_verif["nyquist_criterion_ok"]
    ax.set_xlabel("실수부"); ax.set_ylabel("허수부")
    ax.set_title("Nyquist — 둘러싸기 검증")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    r_max = max(3.0, np.percentile(np.abs(L_jw), 90) * 1.2)
    ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
    apply_grid(ax)
    ax.text(0.98, 0.02,
            f"불안정 개루프 극점: {n_unstable}\n"
            f"폐루프 LHP 내: {'예' if nyq_ok else '아니오'}\n"
            f"Nyquist 기준: {'통과' if nyq_ok else '실패'}",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="green" if nyq_ok else "red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (3,0) Monte Carlo Bode
    ax = axes[3, 0]
    if mc_robustness is not None and len(mc_robustness.get("mc_bode_mag", [])) > 0:
        w_mc = mc_robustness["mc_bode_w"]
        for i, mag_dB in enumerate(mc_robustness["mc_bode_mag"]):
            ax.semilogx(w_mc, mag_dB, color="tab:gray", alpha=0.3, lw=0.5,
                        label="섭동 샘플" if i == 0 else None)
        ax.semilogx(w_mc, mc_robustness["nominal_bode_mag"],
                    color="tab:blue", lw=1.5, label="공칭값")
        ax.set_xlabel("주파수 [rad/s]")
        ax.set_ylabel("진폭 [dB]")
        ax.set_title("Monte Carlo Bode (질량 ±10% 섭동)")
        ax.legend(fontsize=8)
        n_trials = len(mc_robustness["mc_bode_mag"])
        ax.text(0.02, 0.02, f"{n_trials}개 섭동 샘플",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "Monte Carlo 데이터 없음",
                transform=ax.transAxes, ha="center")
    apply_grid(ax)

    # (3,1) 폐루프 극점 산점도
    ax = axes[3, 1]
    if mc_robustness is not None and len(mc_robustness.get("mc_cl_poles", [])) > 0:
        for i, poles in enumerate(mc_robustness["mc_cl_poles"]):
            ax.plot(poles.real, poles.imag, ".", color="tab:gray", ms=4, alpha=0.4,
                    label="섭동 극점" if i == 0 else None)
        nom_poles = mc_robustness["nominal_cl_poles"]
        ax.plot(nom_poles.real, nom_poles.imag, "bo", ms=7, mfc="none", mew=1.5,
                label="공칭 극점")
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("실수부")
        ax.set_ylabel("허수부")
        ax.set_title("폐루프 극점 산점도 (Monte Carlo, 질량 ±10%)")
        ax.legend(fontsize=8)
        all_stable = all(np.all(p.real < 0) for p in mc_robustness["mc_cl_poles"])
        ax.text(0.98, 0.02,
                f"전부 LHP 내: {'예(YES)' if all_stable else '아니오(NO)'}",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                color="green" if all_stable else "red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "Monte Carlo 데이터 없음",
                transform=ax.transAxes, ha="center")
    apply_grid(ax)

    return fig


def _make_roa_plots_ko(roa_data, gs_stability, eq_name):
    """한글 제목이 적용된 ROA 분석 figure를 생성한다."""
    from visualization.common.axis_style import apply_grid

    ko = EQUILIBRIUM_KO[eq_name]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    fig.suptitle(
        f"흡인 영역 (ROA) 및 게인 스케줄링 분석 — {eq_name} 평형점 ({ko})",
        fontsize=14, fontweight="bold"
    )

    # (0,0) ROA 산점도: θ₁ vs θ₂
    ax = axes[0, 0]
    mask = roa_data["converged_mask"]
    t1 = roa_data["theta1_devs"]
    t2 = roa_data["theta2_devs"]
    ax.scatter(t1[~mask], t2[~mask], c="red", s=15, alpha=0.5,
               label="발산", edgecolors="none")
    ax.scatter(t1[mask], t2[mask], c="green", s=15, alpha=0.5,
               label="수렴", edgecolors="none")
    ax.set_xlabel("θ₁ 편차 [deg]")
    ax.set_ylabel("θ₂ 편차 [deg]")
    ax.set_title("흡인 영역 (ROA) — θ₁ vs θ₂")
    ax.legend(fontsize=8)
    apply_grid(ax)
    rate = roa_data["success_rate"]
    ax.text(0.02, 0.98,
            f"수렴 성공률: {rate*100:.1f}%\n"
            f"최대 안정 편차: {roa_data['max_stable_deviation_deg']:.1f} deg",
            transform=ax.transAxes, fontsize=8, ha="left", va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (0,1) 편차별 수렴 성공률 막대 그래프
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
    colors_bar = ["green" if r > 50 else "orange" if r > 20 else "red"
                  for r in bin_rates]
    ax.bar(bin_centers, bin_rates,
           width=(bin_edges[1] - bin_edges[0]) * 0.8,
           color=colors_bar, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("|θ₁| 편차 [deg]")
    ax.set_ylabel("성공률 [%]")
    ax.set_title("초기 편차별 수렴 성공률")
    ax.set_ylim(0, 105)
    apply_grid(ax)

    # (1,0) 운영점별 폐루프 고유값 실수부
    ax = axes[1, 0]
    op_deg = gs_stability["operating_points_deg"]
    eig_list = gs_stability["eigenvalues_per_point"]
    for i, eigs in enumerate(eig_list):
        x_vals = np.full(len(eigs), op_deg[i])
        ax.scatter(x_vals, eigs.real, c="tab:blue", s=20, alpha=0.6,
                   edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1, label="안정성 경계")
    ax.set_xlabel("운영점 편차 [deg]")
    ax.set_ylabel("Re(고유값)")
    ax.set_title("폐루프 고유값 실수부 (게인 스케줄링)")
    ax.legend(fontsize=8)
    apply_grid(ax)
    stable_label = "예(YES)" if gs_stability["all_points_stable"] else "아니오(NO)"
    ax.text(0.98, 0.98,
            f"전 운영점 안정: {stable_label}\n"
            f"최대 Re(고유값): {gs_stability['max_eigenvalue_real']:.4f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color="green" if gs_stability["all_points_stable"] else "red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (1,1) P 행렬 조건수
    ax = axes[1, 1]
    cond = gs_stability["condition_numbers"]
    ax.semilogy(op_deg, cond, "o-", color="tab:purple", lw=1.5, ms=6)
    ax.set_xlabel("운영점 편차 [deg]")
    ax.set_ylabel("P 행렬 조건수")
    ax.set_title("Riccati P 조건수 (강건성 지표)")
    apply_grid(ax)
    ax.text(0.98, 0.98,
            f"최소: {np.min(cond):.1f}\n최대: {np.max(cond):.1f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    return fig


def _make_comparison_plots_ko(comparison_data, eq_name):
    """한글 제목이 적용된 제어기 비교 figure를 생성한다."""
    from visualization.common.axis_style import apply_grid, apply_zero_line

    ko = EQUILIBRIUM_KO[eq_name]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
    fig.suptitle(
        f"제어기 비교: LQR vs PD vs 극점 배치 — {eq_name} 평형점 ({ko})",
        fontsize=14, fontweight="bold"
    )

    name_ko = {"LQR": "LQR", "PD": "PD", "Pole Placement": "극점 배치"}
    colors = {"LQR": "tab:blue", "PD": "tab:orange", "Pole Placement": "tab:green"}
    linestyles = {"LQR": "-", "PD": "--", "Pole Placement": "-."}

    # (0,0) 카트 위치 임펄스 응답
    ax = axes[0, 0]
    for name, data in comparison_data.items():
        if data["is_stable"]:
            ax.plot(data["t"], data["x"][:, 0], color=colors[name],
                    ls=linestyles[name], lw=1.2, label=name_ko.get(name, name))
        else:
            ax.text(0.5, 0.3, f"{name_ko.get(name, name)}: 불안정",
                    transform=ax.transAxes, color=colors[name],
                    fontsize=10, ha="center")
    ax.set_xlabel("시간 [s]")
    ax.set_ylabel("카트 위치 [m]")
    ax.set_title("임펄스 응답: 카트 위치")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (0,1) θ₁ 각도 편차 임펄스 응답
    ax = axes[0, 1]
    for name, data in comparison_data.items():
        if data["is_stable"]:
            ax.plot(data["t"], np.degrees(data["x"][:, 1]),
                    color=colors[name], ls=linestyles[name], lw=1.2,
                    label=name_ko.get(name, name))
    ax.set_xlabel("시간 [s]")
    ax.set_ylabel("θ₁ 편차 [deg]")
    ax.set_title("임펄스 응답: 링크1 각도 편차")
    ax.legend(fontsize=8)
    apply_grid(ax); apply_zero_line(ax)

    # (1,0) 개루프 Bode 진폭 비교
    ax = axes[1, 0]
    for name, data in comparison_data.items():
        ax.semilogx(data["w"], data["L_mag"], color=colors[name],
                    ls=linestyles[name], lw=1.2, label=name_ko.get(name, name))
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("주파수 [rad/s]")
    ax.set_ylabel("진폭 [dB]")
    ax.set_title("개루프 Bode 진폭 비교")
    ax.legend(fontsize=8)
    apply_grid(ax)

    # (1,1) 폐루프 극점 지도 비교
    ax = axes[1, 1]
    markers = {"LQR": "o", "PD": "s", "Pole Placement": "^"}
    for name, data in comparison_data.items():
        eigs = data["eigenvalues"]
        ax.plot(eigs.real, eigs.imag, markers.get(name, "o"),
                color=colors[name],
                ms=7, mfc="none" if name == "LQR" else colors[name],
                mew=1.5, alpha=0.8, label=name_ko.get(name, name))
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    info_lines = []
    for name, data in comparison_data.items():
        status = "안정" if data["is_stable"] else "불안정"
        max_re = np.max(data["eigenvalues"].real)
        info_lines.append(f"{name_ko.get(name, name)}: {status} (max Re={max_re:.2f})")
    ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax.set_xlabel("실수부")
    ax.set_ylabel("허수부")
    ax.set_title("폐루프 극점 지도 비교")
    ax.legend(fontsize=8)
    apply_grid(ax)

    return fig


# ────────────────────────────────────────────────
#  요약 그리드 생성
# ────────────────────────────────────────────────

def _make_summary_grid(results: list[dict]) -> plt.Figure:
    """8개 평형점을 4x2 그리드로 비교하는 summary_grid.png를 생성한다.

    각 셀에 dynamics_analysis.png 썸네일 + 주요 지표를 표시한다.

    Parameters
    ----------
    results : list[dict]
        _run_single_equilibrium 반환값 리스트.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style()
    names = list(EQUILIBRIUM_CONFIGS.keys())  # DDD, DDU, ..., UUU
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), tight_layout=True)
    fig.suptitle("8개 평형점 동역학 분석 요약 비교", fontsize=16, fontweight="bold")

    result_map = {r["eq_name"]: r for r in results}

    for idx, eq_name in enumerate(names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        res = result_map.get(eq_name, {})

        # 이미지 로드 시도
        img_path = os.path.join(IMAGES_DIR, eq_name, "dynamics_analysis.png")
        if os.path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img, aspect="auto")
            except Exception:
                ax.set_facecolor("#F0F0F0")
                ax.text(0.5, 0.5, "이미지 로드 실패",
                        transform=ax.transAxes, ha="center", fontsize=10)
        else:
            ax.set_facecolor("#F5F5F5")
            ax.text(0.5, 0.6, "이미지 없음",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color="gray")

        ax.axis("off")

        # 헤더 텍스트
        ko_name = EQUILIBRIUM_KO.get(eq_name, eq_name)
        stability_ko = EQUILIBRIUM_STABILITY.get(eq_name, "")

        if res.get("success", False):
            roa_rate = res.get("roa_rate", 0.0) * 100
            max_dev = res.get("max_dev_deg", 0.0)
            ts = res.get("settling_time", 0.0)
            cl_ok = res.get("cl_stable", False)
            color = "green" if cl_ok else "red"
            info = (f"ROA: {roa_rate:.0f}%  |  최대 편차: {max_dev:.1f}°\n"
                    f"정착 시간: {ts:.2f} s  |  {stability_ko}")
        else:
            color = "red"
            info = f"실패: {res.get('error', '알 수 없음')[:40]}"

        ax.set_title(
            f"{eq_name}  ({ko_name})",
            fontsize=11, fontweight="bold", color=color, pad=4
        )
        ax.text(0.5, -0.03, info, transform=ax.transAxes,
                ha="center", va="top", fontsize=8,
                color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="lightgray", alpha=0.8))

    return fig


# ────────────────────────────────────────────────
#  메인 진입점
# ────────────────────────────────────────────────

def run_all_equilibria(
    base_cfg: SystemConfig,
    t_end: float = T_END,
    dt: float = DT,
    impulse: float = IMPULSE,
    dist_amplitude: float = DIST_AMPLITUDE,
    dist_bandwidth: float = DIST_BANDWIDTH,
    seed: int = SEED,
    u_max: float = U_MAX,
    equilibria: list[str] | None = None,
) -> list[dict]:
    """8개 평형점(또는 지정 목록) 전체에 대해 시각화를 자동 생성한다.

    Parameters
    ----------
    base_cfg : SystemConfig
        기본 시스템 파라미터.
    t_end, dt, impulse, dist_amplitude, dist_bandwidth, seed, u_max :
        시뮬레이션 파라미터 (pipeline.defaults 기본값 사용).
    equilibria : list[str] or None
        처리할 평형점 목록. None이면 8개 모두 처리.

    Returns
    -------
    list[dict]
        각 평형점의 처리 결과 목록.
    """
    from simulation.warmup import warmup_jit

    if equilibria is None:
        equilibria = list(EQUILIBRIUM_CONFIGS.keys())

    log.info("JIT 함수 워밍업 중...")
    warmup_jit()

    total_start = time.time()
    results = []

    for eq_name in equilibria:
        result = _run_single_equilibrium(
            base_cfg, eq_name,
            t_end=t_end, dt=dt, impulse=impulse,
            dist_amplitude=dist_amplitude,
            dist_bandwidth=dist_bandwidth,
            seed=seed, u_max=u_max,
        )
        results.append(result)
        plt.close("all")  # 메모리 해제

    # 요약 그리드 생성
    log.info("요약 그리드 (summary_grid.png) 생성 중...")
    try:
        fig_summary = _make_summary_grid(results)
        save_figure(fig_summary, "summary_grid", subdir=None)
        plt.close(fig_summary)
    except Exception as exc:
        log.error("요약 그리드 생성 실패: %s", exc, exc_info=True)

    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in results if r.get("success", False))
    log.info("=" * 60)
    log.info("전체 완료: %d / %d 성공, 총 %.1f 초",
             success_count, len(equilibria), total_elapsed)
    log.info("=" * 60)

    return results
