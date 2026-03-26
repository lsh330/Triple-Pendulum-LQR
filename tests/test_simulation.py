"""Tests for simulation loop and integration."""

import numpy as np
import pytest
from parameters.config import SystemConfig


@pytest.fixture
def cfg():
    return SystemConfig(mc=2.4, m1=1.323, m2=1.389, m3=0.8655,
                        L1=0.402, L2=0.332, L3=0.720)


@pytest.fixture
def K(cfg):
    from control.lqr import compute_lqr_gains
    K, _, _, _, _, _ = compute_lqr_gains(cfg)
    return K


class TestSimulationLoop:
    def test_output_shapes(self, cfg, K):
        from simulation.loop.time_loop import simulate
        t, q, dq, u_ctrl, u_dist = simulate(cfg, K, t_end=0.1, dt=0.001)
        N = len(t)
        assert q.shape == (N, 4)
        assert dq.shape == (N, 4)
        assert u_ctrl.shape == (N,)
        assert u_dist.shape == (N,)

    def test_stability(self, cfg, K):
        """System should stay near equilibrium with small impulse."""
        from simulation.loop.time_loop import simulate
        t, q, dq, _, _ = simulate(cfg, K, t_end=3.0, dt=0.001, impulse=1.0)
        # Cart should stay within 1m
        assert np.max(np.abs(q[:, 0])) < 1.0
        # Angles should stay within 30 degrees of equilibrium
        q_eq = cfg.equilibrium
        for j in range(1, 4):
            max_dev = np.max(np.abs(q[:, j] - q_eq[j]))
            assert max_dev < np.deg2rad(30), f"Joint {j} deviation too large: {np.degrees(max_dev):.1f} deg"

    def test_no_nan(self, cfg, K):
        from simulation.loop.time_loop import simulate
        t, q, dq, u_ctrl, u_dist = simulate(cfg, K, t_end=1.0, dt=0.001, impulse=5.0)
        assert not np.any(np.isnan(q))
        assert not np.any(np.isnan(dq))
        assert not np.any(np.isnan(u_ctrl))

    def test_saturation(self, cfg, K):
        """Control force should respect saturation limit."""
        from simulation.loop.time_loop import simulate
        u_max = 100.0
        _, _, _, u_ctrl, _ = simulate(cfg, K, t_end=0.5, dt=0.001,
                                       impulse=5.0, u_max=u_max)
        assert np.max(np.abs(u_ctrl)) <= u_max + 1e-10


class TestImpulseResponse:
    def test_impulse_gives_velocity(self, cfg):
        from simulation.initial_conditions.impulse_response import apply_impulse
        p = cfg.pack()
        q_eq = cfg.equilibrium
        dq0 = apply_impulse(q_eq, p, 5.0)
        assert abs(dq0[0]) > 0, "Cart should have nonzero initial velocity"

    def test_zero_impulse(self, cfg):
        from simulation.initial_conditions.impulse_response import apply_impulse
        p = cfg.pack()
        q_eq = cfg.equilibrium
        dq0 = apply_impulse(q_eq, p, 0.0)
        np.testing.assert_allclose(dq0, 0.0, atol=1e-14)


class TestDisturbance:
    def test_disturbance_shape(self):
        from simulation.disturbance.generate_disturbance import generate_disturbance
        t = np.linspace(0, 1, 1001)
        d = generate_disturbance(t, amplitude=10.0, bandwidth=5.0)
        assert d.shape == t.shape

    def test_disturbance_rms(self):
        from simulation.disturbance.generate_disturbance import generate_disturbance
        t = np.linspace(0, 10, 10001)
        target_rms = 15.0
        d = generate_disturbance(t, amplitude=target_rms, bandwidth=3.0)
        actual_rms = np.sqrt(np.mean(d**2))
        assert abs(actual_rms - target_rms) < 1.0, f"RMS {actual_rms:.1f} != {target_rms}"


class TestGainScheduledSimulation:
    def test_gs_simulation_stable(self, cfg, K):
        """Gain-scheduled simulation should stay stable."""
        from simulation.loop.time_loop import simulate
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        t, q, dq, u_ctrl, _ = simulate(cfg, K, t_end=2.0, dt=0.001,
                                         impulse=3.0, gain_scheduler=gs)
        assert not np.any(np.isnan(q))
        q_eq = cfg.equilibrium
        max_dev = np.max(np.abs(q[-1, 1] - q_eq[1]))
        assert max_dev < np.deg2rad(20), "GS simulation should converge"

    def test_gs_vs_fixed_both_stable(self, cfg, K):
        """Both fixed and gain-scheduled should stabilize."""
        from simulation.loop.time_loop import simulate
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        _, q_gs, _, _, _ = simulate(cfg, K, t_end=1.0, dt=0.001,
                                     impulse=2.0, gain_scheduler=gs)
        _, q_fix, _, _, _ = simulate(cfg, K, t_end=1.0, dt=0.001,
                                      impulse=2.0)
        assert not np.any(np.isnan(q_gs))
        assert not np.any(np.isnan(q_fix))


class TestROA:
    def test_roa_basic(self, cfg, K):
        """ROA estimation should return valid results."""
        from analysis.region_of_attraction import estimate_roa
        roa = estimate_roa(cfg, K, n_samples=20, max_angle_deg=15,
                          t_horizon=1.0, max_samples=20)
        assert 0.0 <= roa['success_rate'] <= 1.0
        assert roa['total_samples'] == 20
        assert len(roa['converged']) == 20
