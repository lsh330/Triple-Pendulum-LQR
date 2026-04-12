"""Tests for LQR control: Riccati solution, gain properties, closed-loop stability."""

import numpy as np
import pytest
from parameters.config import SystemConfig


@pytest.fixture
def cfg():
    return SystemConfig(mc=2.4, m1=1.323, m2=1.389, m3=0.8655,
                        L1=0.402, L2=0.332, L3=0.720)


@pytest.fixture
def lqr_data(cfg):
    from control.lqr import compute_lqr_gains
    return compute_lqr_gains(cfg)


class TestRiccatiSolution:
    def test_P_positive_definite(self, lqr_data):
        _, _, _, P, _, _ = lqr_data
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs > 0), f"P not positive definite: min eig = {eigs.min()}"

    def test_P_symmetric(self, lqr_data):
        _, _, _, P, _, _ = lqr_data
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_riccati_equation(self, lqr_data):
        """P should satisfy A'P + PA - PBR^{-1}B'P + Q = 0."""
        K, A, B, P, Q, R = lqr_data
        residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
        np.testing.assert_allclose(residual, 0.0, atol=1e-8,
                                   err_msg="Riccati equation not satisfied")


class TestGainMatrix:
    def test_K_shape(self, lqr_data):
        K, _, _, _, _, _ = lqr_data
        assert K.shape == (1, 8)

    def test_K_formula(self, lqr_data):
        """K should equal R^{-1} B' P."""
        K, _, B, P, _, R = lqr_data
        K_expected = np.linalg.solve(R, B.T @ P)
        np.testing.assert_allclose(K, K_expected, atol=1e-10)


class TestClosedLoopStability:
    def test_all_poles_lhp(self, lqr_data):
        K, A, B, _, _, _ = lqr_data
        from control.closed_loop import compute_closed_loop
        cl = compute_closed_loop(A, B, K)
        assert cl["is_stable"], "Closed-loop system should be stable"
        max_re = np.max(cl["poles"].real)
        assert max_re < -0.1, f"Dominant pole too close to imaginary axis: Re={max_re:.4f}"

    def test_return_difference(self, lqr_data):
        """Kalman inequality: |1 + L(jw)| >= 1 for all w."""
        K, A, B, _, _, _ = lqr_data
        import scipy.signal as sig

        C_L = K.reshape(1, -1)
        D_L = np.zeros((1, 1))
        sys_L = sig.lti(A, B.reshape(-1, 1), C_L, D_L)
        w = np.logspace(-2, 3, 1000)
        _, H = sig.freqresp(sys_L, w=w)
        return_diff = np.abs(1.0 + H.flatten())
        assert np.all(return_diff >= 1.0 - 1e-6), \
            f"Kalman inequality violated: min |1+L| = {return_diff.min()}"


class TestGainScheduling:
    def test_scheduler_creation(self, cfg):
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        assert gs.K_gains.shape[1] == 8
        assert len(gs.deviation_angles) == 7

    def test_interpolation_at_zero(self, cfg):
        """Interpolated gain at zero deviation should match nominal LQR."""
        from control.gain_scheduling import GainScheduler
        from control.lqr import compute_lqr_gains
        gs = GainScheduler(cfg)
        K_nom, _, _, _, _, _ = compute_lqr_gains(cfg)
        K_interp = gs.get_gain(cfg.equilibrium, cfg.equilibrium)
        np.testing.assert_allclose(K_interp, K_nom.flatten(), atol=1e-6)

    def test_pack_for_njit(self, cfg):
        from control.gain_scheduling import GainScheduler
        gs = GainScheduler(cfg)
        dev, K, slopes = gs.pack_for_njit()
        assert dev.shape == (7,)
        assert K.shape == (7, 8)
        assert slopes.shape == (7, 8)
