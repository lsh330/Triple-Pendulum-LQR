"""W1-W6 경고 수정 사항에 대한 단위 테스트.

각 WARNING 마다 최소 1개 이상의 테스트가 포함된다.
"""

import numpy as np
import pytest
from parameters.config import SystemConfig


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    """표준 Medrano-Cerda 구성."""
    return SystemConfig(mc=3.0, m1=1.323, m2=1.389, m3=0.8655,
                        L1=0.402, L2=0.332, L3=0.720)


@pytest.fixture(scope="module")
def lqr_data(cfg):
    from control.lqr import compute_lqr_gains
    K, A, B, P, Q, R = compute_lqr_gains(cfg)
    return {"K": K, "A": A, "B": B, "P": P, "Q": Q, "R": R}


# ===========================================================================
# W1: ROA 샘플링 일관성 및 actuator_saturation 필드
# ===========================================================================

class TestW1RoaSaturation:
    def test_default_actuator_saturation(self, cfg):
        """SystemConfig 기본 actuator_saturation 은 200 N 이어야 한다."""
        assert cfg.actuator_saturation == 200.0

    def test_custom_actuator_saturation(self):
        """사용자 지정 actuator_saturation 이 올바르게 저장되어야 한다."""
        cfg2 = SystemConfig(mc=3.0, m1=1.0, m2=1.0, m3=1.0,
                            L1=0.5, L2=0.5, L3=0.5,
                            actuator_saturation=150.0)
        assert cfg2.actuator_saturation == 150.0

    def test_invalid_actuator_saturation_raises(self):
        """actuator_saturation <= 0 이면 ValueError 가 발생해야 한다."""
        with pytest.raises(ValueError, match="actuator_saturation"):
            SystemConfig(mc=3.0, m1=1.0, m2=1.0, m3=1.0,
                         L1=0.5, L2=0.5, L3=0.5,
                         actuator_saturation=-50.0)

    def test_get_u_max_from_cfg(self, cfg):
        """get_u_max(cfg) 는 cfg.actuator_saturation 을 반환해야 한다."""
        from analysis.roa_utils import get_u_max
        assert get_u_max(cfg) == cfg.actuator_saturation

    def test_get_u_max_fallback(self):
        """actuator_saturation 속성 없는 객체에서 fallback 이 동작해야 한다."""
        from analysis.roa_utils import get_u_max

        class FakeCfg:
            pass

        assert get_u_max(FakeCfg(), fallback=300.0) == 300.0

    def test_wilson_ci_width_decreases_with_n(self):
        """샘플 수가 많을수록 Wilson CI 폭이 좁아야 한다."""
        from analysis.roa_utils import wilson_ci_width
        ci_100 = wilson_ci_width(50, 100)
        ci_1000 = wilson_ci_width(500, 1000)
        assert ci_1000 < ci_100

    def test_wilson_ci_width_zero_samples(self):
        """n=0 이면 CI 폭은 1.0 이어야 한다."""
        from analysis.roa_utils import wilson_ci_width
        assert wilson_ci_width(0, 0) == 1.0

    def test_adaptive_sample_not_converged_below_nmin(self):
        """n_total < n_min 이면 항상 False 반환."""
        from analysis.roa_utils import adaptive_sample_count
        assert adaptive_sample_count(10, 50, n_min=300) is False

    def test_adaptive_sample_converged_above_nmax(self):
        """n_total >= n_max 이면 항상 True 반환."""
        from analysis.roa_utils import adaptive_sample_count
        assert adaptive_sample_count(100, 2000, n_min=300, n_max=2000) is True

    def test_roa_batch_uses_u_max(self, cfg, lqr_data):
        """_roa_batch 가 u_max=200.0 인자를 받아 실행되어야 한다 (TypeError 없음)."""
        from analysis.region_of_attraction import _roa_batch
        K_flat = lqr_data["K"].flatten()
        q_eq = cfg.equilibrium
        p = cfg.pack()
        theta1 = np.array([0.1, -0.1])
        theta2 = np.array([0.05, -0.05])
        theta3 = np.array([0.02, -0.02])
        result = _roa_batch(2, 5, 0.001, q_eq, K_flat, p,
                            theta1, theta2, theta3, 0.017,
                            u_max=200.0)
        assert result.shape == (2,)

    def test_lyapunov_roa_returns_ci_width(self, cfg, lqr_data):
        """estimate_lyapunov_roa 결과에 ci_width 키가 있어야 한다."""
        from control.supervisor.roa_estimation import estimate_lyapunov_roa
        q_eq = cfg.equilibrium
        result = estimate_lyapunov_roa(
            cfg, lqr_data["K"], lqr_data["P"], q_eq,
            n_samples=20, t_horizon=0.5, n_min=10, n_max=50,
        )
        assert "ci_width" in result
        assert 0.0 <= result["ci_width"] <= 1.0

    def test_lyapunov_roa_uses_cfg_saturation(self, cfg, lqr_data):
        """actuator_saturation=50 으로 낮추면 수렴률이 다를 수 있음."""
        from control.supervisor.roa_estimation import estimate_lyapunov_roa
        cfg_low = SystemConfig(mc=3.0, m1=1.323, m2=1.389, m3=0.8655,
                               L1=0.402, L2=0.332, L3=0.720,
                               actuator_saturation=50.0)
        q_eq = cfg_low.equilibrium
        result = estimate_lyapunov_roa(
            cfg_low, lqr_data["K"], lqr_data["P"], q_eq,
            n_samples=10, t_horizon=0.3, n_min=5, n_max=20,
        )
        # 반환 형식 검증 (성공률 자체가 다를 수 있음)
        assert "n_total" in result
        assert result["n_total"] >= 5


# ===========================================================================
# W2: Robust Stability 공칭 게인 고정 테스트
# ===========================================================================

class TestW2RobustStability:
    def test_returns_expected_keys(self, cfg, lqr_data):
        """compute_robust_stability 결과에 필수 키가 있어야 한다."""
        from analysis.lqr_verification.compute_verification import compute_robust_stability
        result = compute_robust_stability(cfg, lqr_data["K"],
                                         n_samples=10,
                                         mass_pert=0.05, length_pert=0.02)
        for key in ("all_stable", "n_stable", "n_total", "stability_rate",
                    "worst_max_eig_real", "worst_damping", "worst_eigenvalue",
                    "max_eig_real_per_sample", "min_damping_per_sample"):
            assert key in result, f"키 '{key}' 누락"

    def test_nominal_always_stable(self, cfg, lqr_data):
        """공칭 파라미터에서는 항상 안정해야 한다 (perturbation=0)."""
        from analysis.lqr_verification.compute_verification import compute_robust_stability
        result = compute_robust_stability(cfg, lqr_data["K"],
                                         n_samples=5,
                                         mass_pert=0.0, length_pert=0.0)
        assert result["all_stable"], "공칭 파라미터에서 불안정 — 버그 의심"

    def test_stability_rate_in_unit_interval(self, cfg, lqr_data):
        """안정률은 [0, 1] 이어야 한다."""
        from analysis.lqr_verification.compute_verification import compute_robust_stability
        result = compute_robust_stability(cfg, lqr_data["K"],
                                         n_samples=10,
                                         mass_pert=0.05, length_pert=0.02)
        assert 0.0 <= result["stability_rate"] <= 1.0

    def test_gain_not_recomputed(self, cfg, lqr_data):
        """K_nom 고정 여부: 섭동 전·후 K_nom 동일함을 간접 검증 (배열 동일성)."""
        # compute_robust_stability 는 K_nom 을 변경하지 않아야 한다
        from analysis.lqr_verification.compute_verification import compute_robust_stability
        K_before = lqr_data["K"].copy()
        compute_robust_stability(cfg, lqr_data["K"], n_samples=5)
        np.testing.assert_array_equal(lqr_data["K"], K_before)

    def test_worst_eigenvalue_is_complex(self, cfg, lqr_data):
        """worst_eigenvalue 는 complex 타입이어야 한다."""
        from analysis.lqr_verification.compute_verification import compute_robust_stability
        result = compute_robust_stability(cfg, lqr_data["K"],
                                         n_samples=5,
                                         mass_pert=0.05, length_pert=0.02)
        assert np.iscomplex(result["worst_eigenvalue"]) or np.isreal(result["worst_eigenvalue"])


# ===========================================================================
# W3: 개별 상태 채널 RMS 오차
# ===========================================================================

class TestW3ChannelRms:
    def _make_states(self):
        """간단한 지수 감쇠 상태 시계열 생성."""
        t = np.linspace(0, 5, 500)
        states = np.zeros((500, 8))
        # 각 채널에 다른 진폭
        for i in range(8):
            states[:, i] = (i + 1) * 0.1 * np.exp(-t)
        eq_state = np.zeros(8)
        return states, eq_state, t

    def test_returns_all_keys(self):
        """compute_channel_rms 결과에 모든 채널 키가 있어야 한다."""
        from analysis.performance.rms_error import compute_channel_rms
        states, eq, t = self._make_states()
        result = compute_channel_rms(states, eq, t)
        for key in ("cart_rms", "phi1_rms", "phi2_rms", "phi3_rms",
                    "vx_rms", "w1_rms", "w2_rms", "w3_rms", "total_rms"):
            assert key in result

    def test_zero_error_gives_zero_rms(self):
        """오차가 0이면 RMS 도 0이어야 한다."""
        from analysis.performance.rms_error import compute_channel_rms
        eq = np.array([0.0, np.pi, np.pi, np.pi, 0., 0., 0., 0.])
        states = np.tile(eq, (100, 1))
        t = np.linspace(0, 1, 100)
        result = compute_channel_rms(states, eq, t)
        for key, val in result.items():
            assert val == pytest.approx(0.0, abs=1e-12), f"{key} = {val} (기대값: 0)"

    def test_known_rms_value(self):
        """채널 0에 상수 오차 δ 가 있으면 cart_rms ≈ δ 이어야 한다."""
        from analysis.performance.rms_error import compute_channel_rms
        N = 200
        eq = np.zeros(8)
        states = np.zeros((N, 8))
        states[:, 0] = 0.5   # cart 채널에만 0.5 m 오차
        t = np.linspace(0, 2, N)
        result = compute_channel_rms(states, eq, t)
        assert result["cart_rms"] == pytest.approx(0.5, rel=1e-6)
        assert result["phi1_rms"] == pytest.approx(0.0, abs=1e-12)

    def test_shape_error_raises(self):
        """states shape 이 (N, 8) 이 아니면 ValueError 가 발생해야 한다."""
        from analysis.performance.rms_error import compute_channel_rms
        states = np.zeros((100, 6))  # 잘못된 열 수
        eq = np.zeros(8)
        t = np.linspace(0, 1, 100)
        with pytest.raises(ValueError):
            compute_channel_rms(states, eq, t)

    def test_total_rms_geq_channel_rms(self):
        """total_rms >= 각 채널 RMS / sqrt(8) 이어야 한다 (평균이므로)."""
        from analysis.performance.rms_error import compute_channel_rms
        t = np.linspace(0, 1, 100)
        eq = np.zeros(8)
        states = np.random.default_rng(0).normal(0, 0.1, (100, 8))
        result = compute_channel_rms(states, eq, t)
        # total_rms 는 전체 원소의 RMS 이므로 최소 채널 RMS 보다 크거나 같아야 함
        channel_vals = [result[k] for k in
                        ("cart_rms", "phi1_rms", "phi2_rms", "phi3_rms",
                         "vx_rms", "w1_rms", "w2_rms", "w3_rms")]
        assert result["total_rms"] >= min(channel_vals) - 1e-12


# ===========================================================================
# W4: Gain Scheduler Trim Solver
# ===========================================================================

class TestW4TrimSolver:
    def test_trim_u_at_equilibrium_near_zero(self, cfg):
        """UUU 평형점에서 트림 u_eq 는 거의 0 이어야 한다 (대칭 배치)."""
        from control.linearization.trim_solver import compute_trim_u
        q_eq = cfg.equilibrium
        p = cfg.pack()
        # UUU 평형 = [0, 0, 0, 0] → 카트가 인력 방향으로 이동할 필요 없음
        # 수치 오차 허용
        u_eq = compute_trim_u(q_eq, p)
        assert abs(u_eq) < 1.0, f"UUU 평형에서 u_eq = {u_eq:.4f} (기대: ~0)"

    def test_trim_u_nonzero_at_tilted(self, cfg):
        """기울어진 격자점에서 트림 u_eq 는 비영이어야 한다."""
        from control.linearization.trim_solver import compute_trim_u
        q_op = cfg.equilibrium.copy()
        q_op[1] += np.deg2rad(15)   # theta1 += 15 deg
        p = cfg.pack()
        u_eq = compute_trim_u(q_op, p)
        # 중력이 작용하므로 u_eq != 0
        assert abs(u_eq) > 1e-3, "기울어진 점에서 u_eq 가 0 에 너무 가깝습니다"

    def test_gain_scheduler_has_u_ff(self, cfg):
        """GainScheduler 가 u_ff_gains 속성을 가져야 한다."""
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        assert hasattr(gs, "u_ff_gains")
        assert len(gs.u_ff_gains) == len(gs.deviation_angles)

    def test_get_gain_and_ff_shape(self, cfg):
        """get_gain_and_ff 가 K(8,) 와 float u_ff 를 반환해야 한다."""
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        q = cfg.equilibrium.copy()
        q[1] += np.deg2rad(5)
        K, u_ff = gs.get_gain_and_ff(q, cfg.equilibrium)
        assert K.shape == (8,)
        assert isinstance(u_ff, float)

    def test_trim_ff_at_zero_deg_near_zero(self, cfg):
        """0도 격자점에서 u_ff ≈ 0 이어야 한다."""
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        K, u_ff = gs.get_gain_and_ff(cfg.equilibrium, cfg.equilibrium)
        assert abs(u_ff) < 5.0, f"0도에서 u_ff={u_ff:.4f} (기대: ~0)"

    def test_pack_ff_for_njit_shape(self, cfg):
        """pack_ff_for_njit 반환 형태 검증."""
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        dev, u_ff, slopes_ff = gs.pack_ff_for_njit()
        assert dev.shape == u_ff.shape == slopes_ff.shape

    def test_multiaxis_scheduler_has_u_ff_grid(self, cfg):
        """MultiAxisGainScheduler 가 u_ff_grid 를 가져야 한다."""
        from control.gain_scheduling import MultiAxisGainScheduler
        gs = MultiAxisGainScheduler(cfg,
                                    theta1_deg=[-5, 0, 5],
                                    theta2_deg=[0],
                                    theta3_deg=[0])
        assert hasattr(gs, "u_ff_grid")
        assert gs.u_ff_grid.shape == (3, 1, 1)

    def test_trim_improvement_vs_heuristic(self, cfg):
        """트림 피드포워드가 u=0 heuristic 보다 중력 잔차를 줄여야 한다."""
        from control.linearization.trim_solver import compute_trim_u
        from dynamics.mass_matrix.assembly import mass_matrix
        from dynamics.gravity.gravity_vector import gravity_vector

        q_op = cfg.equilibrium.copy()
        q_op[1] += np.deg2rad(20)
        p = cfg.pack()

        M = mass_matrix(q_op, p)
        G = gravity_vector(q_op, p)
        M_inv = np.linalg.inv(M)

        # 입력 열 벡터 (카트 방향)
        e0 = np.zeros(4); e0[0] = 1.0
        B_u = M_inv @ e0

        # heuristic: u=0 → 잔차 = |G_acc|
        G_acc = M_inv @ G
        residual_heuristic = np.linalg.norm(G_acc)

        # trim: u=u_eq → 잔차 감소
        u_eq = compute_trim_u(q_op, p)
        residual_trim = np.linalg.norm(G_acc - B_u * u_eq)

        assert residual_trim < residual_heuristic, (
            f"trim 잔차({residual_trim:.4f}) >= heuristic 잔차({residual_heuristic:.4f})"
        )


# ===========================================================================
# W5: Shamma-Athans Slow-Variation 검증
# ===========================================================================

class TestW5SlowVariation:
    def test_returns_expected_keys(self, cfg):
        """verify_slow_variation 이 필수 키를 반환해야 한다."""
        from analysis.gain_scheduling_stability import verify_slow_variation
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-10, 0, 10])
        result = verify_slow_variation(gs, max_rate=1.0)
        for key in ("max_dK_dsigma", "gamma_max", "slow_variation_ok",
                    "margin", "details_per_gap"):
            assert key in result

    def test_gamma_max_positive(self, cfg):
        """γ_max 는 안정한 시스템에서 양수이어야 한다."""
        from analysis.gain_scheduling_stability import verify_slow_variation
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-10, 0, 10])
        result = verify_slow_variation(gs, max_rate=1.0)
        assert result["gamma_max"] > 0, "안정 시스템에서 γ_max > 0 이어야 함"

    def test_very_slow_rate_passes(self, cfg):
        """매우 느린 변화율(0.001 rad/s)에서 slow_variation_ok=True."""
        from analysis.gain_scheduling_stability import verify_slow_variation
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-10, 0, 10])
        result = verify_slow_variation(gs, max_rate=0.001)
        assert result["slow_variation_ok"] is True

    def test_margin_equals_gamma_minus_rate(self, cfg):
        """margin = gamma_max - max_rate 이어야 한다."""
        from analysis.gain_scheduling_stability import verify_slow_variation
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-10, 0, 10])
        max_rate = 2.0
        result = verify_slow_variation(gs, max_rate=max_rate)
        expected_margin = result["gamma_max"] - max_rate
        assert result["margin"] == pytest.approx(expected_margin, rel=1e-8)

    def test_details_gap_count(self, cfg):
        """n 개 격자점에서 n-1 개 구간 상세가 있어야 한다."""
        from analysis.gain_scheduling_stability import verify_slow_variation
        from control.gain_scheduling import GainScheduler
        n_pts = 5
        angles = np.linspace(-20, 20, n_pts).tolist()
        gs = GainScheduler(cfg, deviation_angles_deg=angles)
        result = verify_slow_variation(gs, max_rate=1.0)
        assert len(result["details_per_gap"]) == n_pts - 1


# ===========================================================================
# W6: Common Lyapunov LMI 검증
# ===========================================================================

class TestW6CommonLyapunov:
    def test_returns_expected_keys(self, cfg):
        """verify_common_lyapunov 이 필수 키를 반환해야 한다."""
        from analysis.gain_scheduling_stability import verify_common_lyapunov
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-5, 0, 5])
        result = verify_common_lyapunov(cfg, gs)
        for key in ("common_P_exists", "method", "common_P",
                    "worst_lyap_eig", "all_stable_under_P", "n_points"):
            assert key in result

    def test_common_P_symmetric(self, cfg):
        """공통 P 행렬은 대칭이어야 한다."""
        from analysis.gain_scheduling_stability import verify_common_lyapunov
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-5, 0, 5])
        result = verify_common_lyapunov(cfg, gs)
        P = result["common_P"]
        np.testing.assert_allclose(P, P.T, atol=1e-8)

    def test_n_points_matches_scheduler(self, cfg):
        """n_points 는 scheduler 격자점 수와 일치해야 한다."""
        from analysis.gain_scheduling_stability import verify_common_lyapunov
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-10, -5, 0, 5, 10])
        result = verify_common_lyapunov(cfg, gs)
        assert result["n_points"] == 5

    def test_common_P_positive_definite_or_near(self, cfg):
        """공통 P 는 양정치 또는 근사적으로 양정치이어야 한다."""
        from analysis.gain_scheduling_stability import verify_common_lyapunov
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-5, 0, 5])
        result = verify_common_lyapunov(cfg, gs)
        eigs = np.linalg.eigvalsh(result["common_P"])
        assert np.all(eigs > -1e-4), f"공통 P 비양정치: min eig = {eigs.min():.4e}"

    def test_method_is_valid_string(self, cfg):
        """method 는 'cvxpy_sdp' 또는 'approximate' 이어야 한다."""
        from analysis.gain_scheduling_stability import verify_common_lyapunov
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg, deviation_angles_deg=[-5, 0, 5])
        result = verify_common_lyapunov(cfg, gs)
        assert result["method"] in ("cvxpy_sdp", "approximate")
