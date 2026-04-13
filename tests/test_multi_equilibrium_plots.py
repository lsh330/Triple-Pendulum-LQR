"""다중 평형점 시각화 생성 테스트.

검증 항목:
  - korean_font.py: 폰트 탐지 및 적용
  - multi_equilibrium_runner: 단일 평형점 파이프라인 실행
  - save_outputs: 서브디렉터리 저장 경로 생성
  - summary_grid: 8개 평형점 요약 그리드 생성
  - main.py --all-equilibria 플래그 파싱
"""

import os
import sys
import types
import tempfile
import numpy as np
import pytest

# ── 공통 픽스처 ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """표준 Medrano-Cerda 구성."""
    from parameters.config import SystemConfig
    return SystemConfig(mc=2.4, m1=1.323, m2=1.389, m3=0.8655,
                        L1=0.402, L2=0.332, L3=0.720)


# ── 한글 폰트 테스트 ─────────────────────────────────────────────────────────

class TestKoreanFont:
    """korean_font.py 모듈 기능 테스트."""

    def test_apply_korean_font_returns_string(self):
        """apply_korean_font()는 문자열 폰트명을 반환해야 한다."""
        from visualization.common.korean_font import apply_korean_font, reset_font
        reset_font()
        result = apply_korean_font()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_apply_korean_font_cached(self):
        """두 번 연속 호출 시 동일한 폰트명을 반환해야 한다 (캐시)."""
        from visualization.common.korean_font import apply_korean_font, reset_font
        reset_font()
        first = apply_korean_font()
        second = apply_korean_font()
        assert first == second

    def test_reset_font_clears_cache(self):
        """reset_font() 후 재호출 시에도 폰트명이 반환된다."""
        from visualization.common.korean_font import apply_korean_font, reset_font
        reset_font()
        first = apply_korean_font()
        reset_font()
        second = apply_korean_font()
        assert isinstance(second, str)

    def test_matplotlib_rcparams_updated(self):
        """apply_korean_font() 적용 후 matplotlib font.family가 변경된다."""
        import matplotlib
        from visualization.common.korean_font import apply_korean_font, reset_font
        reset_font()
        font_name = apply_korean_font()
        # rcParams["font.family"]는 문자열 또는 리스트 형태로 반환될 수 있다
        family = matplotlib.rcParams["font.family"]
        if isinstance(family, list):
            assert font_name in family or family[0] == font_name
        else:
            assert family == font_name
        assert not matplotlib.rcParams["axes.unicode_minus"]


# ── axis_style 테스트 ─────────────────────────────────────────────────────────

class TestAxisStyle:
    """axis_style.py 한글 폰트 통합 테스트."""

    def test_apply_publication_style_runs(self):
        """apply_publication_style()이 예외 없이 실행되어야 한다."""
        from visualization.common.axis_style import apply_publication_style
        apply_publication_style()  # 예외 없으면 통과

    def test_apply_grid_and_zero_line(self):
        """apply_grid, apply_zero_line이 axes 객체에 정상 적용된다."""
        import matplotlib.pyplot as plt
        from visualization.common.axis_style import apply_grid, apply_zero_line
        fig, ax = plt.subplots()
        apply_grid(ax)
        apply_zero_line(ax)
        plt.close(fig)


# ── save_outputs 서브디렉터리 테스트 ─────────────────────────────────────────

class TestSaveOutputsSubdir:
    """save_outputs.py 서브디렉터리 지원 테스트."""

    def test_ensure_dir_creates_subdir(self, tmp_path, monkeypatch):
        """ensure_dir(subdir)가 images/{subdir}/ 를 생성해야 한다."""
        import pipeline.save_outputs as so
        monkeypatch.setattr(so, "IMAGES_DIR", str(tmp_path))
        result = so.ensure_dir("UUU")
        assert os.path.isdir(result)
        assert result.endswith("UUU")

    def test_ensure_dir_none_creates_root(self, tmp_path, monkeypatch):
        """ensure_dir(None)이 root images/ 디렉터리를 생성해야 한다."""
        import pipeline.save_outputs as so
        monkeypatch.setattr(so, "IMAGES_DIR", str(tmp_path))
        result = so.ensure_dir(None)
        assert os.path.isdir(result)
        assert result == str(tmp_path)

    def test_save_figure_subdir(self, tmp_path, monkeypatch):
        """save_figure(fig, name, subdir='DDD')가 images/DDD/name.png를 생성한다."""
        import matplotlib.pyplot as plt
        import pipeline.save_outputs as so
        monkeypatch.setattr(so, "IMAGES_DIR", str(tmp_path))
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        path = so.save_figure(fig, "test_plot", subdir="DDD")
        plt.close(fig)
        assert os.path.isfile(path)
        assert "DDD" in path
        assert path.endswith(".png")


# ── 한글 제목 시각화 함수 테스트 ─────────────────────────────────────────────

class TestKoreanTitlePlots:
    """multi_equilibrium_runner 내 한글 제목 시각화 함수 기본 테스트."""

    @pytest.fixture(scope="class")
    def sim_data(self, cfg):
        """UUU 평형점 시뮬레이션 데이터 픽스처."""
        from control.lqr import compute_lqr_gains
        from control.closed_loop import compute_closed_loop
        from control.gain_scheduling import GainScheduler
        from simulation.disturbance.generate_disturbance import generate_disturbance
        from simulation.loop.time_loop import simulate
        from analysis.state.derived_state import compute_derived_state
        from analysis.energy.total_energy import compute_energy
        from analysis.frequency.frequency_response import compute_frequency_response
        from simulation.warmup import warmup_jit

        warmup_jit()
        cfg.set_equilibrium("UUU")
        K, A, B, P, Q, R = compute_lqr_gains(cfg)
        cl = compute_closed_loop(A, B, K)
        gs = GainScheduler(cfg)

        t_tmp = np.linspace(0, 5.0, 5001)
        dist = generate_disturbance(t_tmp, amplitude=15.0, bandwidth=3.0, seed=42)
        t, q, dq, u_ctrl, u_dist, _, _ = simulate(
            cfg, K, t_end=5.0, dt=0.001, impulse=5.0,
            disturbance=dist, gain_scheduler=gs, u_max=200.0)

        state = compute_derived_state(cfg, q, dq)
        energy = compute_energy(cfg, q, dq,
                                state["phi1"], state["phi2"], state["phi3"])
        freq_data = compute_frequency_response(A, B, K, cl["A_cl"])

        return dict(t=t, q=q, dq=dq, state=state, energy=energy,
                    u_ctrl=u_ctrl, u_dist=u_dist, dt=0.001,
                    K=K, A=A, B=B, P=P, Q=Q, R=R,
                    freq_data=freq_data)

    def test_dynamics_plots_ko(self, cfg, sim_data):
        """_make_dynamics_plots_ko 함수가 Figure를 반환해야 한다."""
        import matplotlib.pyplot as plt
        from pipeline.multi_equilibrium_runner import _make_dynamics_plots_ko
        d = sim_data
        fig = _make_dynamics_plots_ko(
            d["t"], d["q"], d["dq"], d["state"], d["energy"],
            d["u_ctrl"], "UUU"
        )
        assert fig is not None
        assert len(fig.axes) == 10
        plt.close(fig)

    def test_control_plots_ko(self, cfg, sim_data):
        """_make_control_plots_ko 함수가 Figure를 반환해야 한다."""
        import matplotlib.pyplot as plt
        from pipeline.multi_equilibrium_runner import _make_control_plots_ko
        d = sim_data
        fig = _make_control_plots_ko(
            d["t"], d["u_ctrl"], d["u_dist"], d["dt"],
            d["freq_data"], "UUU"
        )
        assert fig is not None
        assert len(fig.axes) >= 8
        plt.close(fig)

    def test_comparison_plots_ko(self, cfg, sim_data):
        """_make_comparison_plots_ko 함수가 Figure를 반환해야 한다."""
        import matplotlib.pyplot as plt
        from control.comparison import compare_controllers
        from pipeline.multi_equilibrium_runner import _make_comparison_plots_ko
        d = sim_data
        comparison_data = compare_controllers(d["A"], d["B"], d["K"])
        fig = _make_comparison_plots_ko(comparison_data, "UUU")
        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)


# ── EQUILIBRIUM_KO 사전 완전성 테스트 ────────────────────────────────────────

class TestEquilibriumMetadata:
    """평형점 메타데이터 완전성 테스트."""

    def test_all_8_configs_in_ko_dict(self):
        """EQUILIBRIUM_KO 사전이 8개 평형점을 모두 포함해야 한다."""
        from parameters.equilibrium import EQUILIBRIUM_CONFIGS
        from pipeline.multi_equilibrium_runner import EQUILIBRIUM_KO
        for name in EQUILIBRIUM_CONFIGS:
            assert name in EQUILIBRIUM_KO, f"{name} 한글 설명 누락"

    def test_equilibrium_ko_values_are_nonempty_strings(self):
        """EQUILIBRIUM_KO 값들이 모두 비어 있지 않은 문자열이어야 한다."""
        from pipeline.multi_equilibrium_runner import EQUILIBRIUM_KO
        for name, desc in EQUILIBRIUM_KO.items():
            assert isinstance(desc, str) and len(desc) > 0, \
                f"{name} 한글 설명이 빈 문자열"


# ── 단일 평형점 실행 스모크 테스트 ────────────────────────────────────────────

class TestSingleEquilibriumSmoke:
    """단일 평형점 _run_single_equilibrium 스모크 테스트 (빠른 설정 사용)."""

    def test_run_single_uuu_success(self, cfg, tmp_path, monkeypatch):
        """UUU 평형점 파이프라인이 성공적으로 완료되고 6개 파일을 저장한다."""
        import pipeline.save_outputs as so
        from pipeline.multi_equilibrium_runner import _run_single_equilibrium

        monkeypatch.setattr(so, "IMAGES_DIR", str(tmp_path))

        result = _run_single_equilibrium(
            cfg, "UUU",
            t_end=3.0, dt=0.001, impulse=3.0,
            dist_amplitude=5.0, dist_bandwidth=3.0,
            seed=42, u_max=200.0,
        )

        assert result["success"], f"실패 원인: {result.get('error', '')}"
        assert result["eq_name"] == "UUU"
        assert result["elapsed"] > 0

        expected_files = [
            "dynamics_analysis.png",
            "control_analysis.png",
            "lqr_verification.png",
            "roa_analysis.png",
            "comparison_analysis.png",
            "animation.gif",
        ]
        uuu_dir = os.path.join(str(tmp_path), "UUU")
        assert os.path.isdir(uuu_dir), "UUU 서브디렉터리가 생성되지 않음"
        for fname in expected_files:
            fpath = os.path.join(uuu_dir, fname)
            assert os.path.isfile(fpath), f"{fname} 파일이 생성되지 않음"
            assert os.path.getsize(fpath) > 0, f"{fname} 파일이 비어 있음"


# ── CLI --all-equilibria 플래그 파싱 테스트 ───────────────────────────────────

class TestCLIAllEquilibriaFlag:
    """main.py --all-equilibria 플래그 파싱 테스트."""

    def test_all_equilibria_flag_parsed(self):
        """--all-equilibria 플래그가 args.all_equilibria=True로 파싱된다."""
        import main as m
        parser = m._build_parser()
        args = parser.parse_args(["--all-equilibria", "--no-display"])
        assert args.all_equilibria is True

    def test_equilibria_list_parsed(self):
        """--equilibria-list 플래그가 올바르게 파싱된다."""
        import main as m
        parser = m._build_parser()
        args = parser.parse_args(["--all-equilibria", "--equilibria-list", "UUU,DDD"])
        assert args.equilibria_list == "UUU,DDD"

    def test_default_all_equilibria_false(self):
        """기본값에서 --all-equilibria 플래그는 False 이어야 한다."""
        import main as m
        parser = m._build_parser()
        args = parser.parse_args([])
        assert args.all_equilibria is False
