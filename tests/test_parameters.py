"""Tests for parameter management."""

import numpy as np
import pytest
from parameters.config import SystemConfig


class TestSystemConfig:
    def test_creation(self):
        cfg = SystemConfig(mc=2.4, m1=1.0, m2=1.0, m3=1.0,
                          L1=0.5, L2=0.5, L3=0.5)
        assert cfg.mc == 2.4
        assert cfg.L1 == 0.5

    def test_validation_negative_mass(self):
        with pytest.raises(ValueError):
            SystemConfig(mc=-1.0, m1=1.0, m2=1.0, m3=1.0,
                        L1=0.5, L2=0.5, L3=0.5)

    def test_validation_zero_length(self):
        with pytest.raises(ValueError):
            SystemConfig(mc=1.0, m1=1.0, m2=1.0, m3=1.0,
                        L1=0.0, L2=0.5, L3=0.5)

    def test_pack_shape(self):
        cfg = SystemConfig(mc=1.0, m1=1.0, m2=1.0, m3=1.0,
                          L1=1.0, L2=1.0, L3=1.0)
        p = cfg.pack()
        assert p.shape == (13,)
        assert p.dtype == np.float64

    def test_equilibrium(self):
        cfg = SystemConfig(mc=1.0, m1=1.0, m2=1.0, m3=1.0,
                          L1=1.0, L2=1.0, L3=1.0)
        q_eq = cfg.equilibrium
        assert q_eq.shape == (4,)
        assert q_eq[0] == 0.0
        assert abs(q_eq[1] - np.pi) < 1e-14
        assert q_eq[2] == 0.0
        assert q_eq[3] == 0.0

    def test_heavy_cart(self):
        """Heavy cart should still produce valid LQR."""
        cfg = SystemConfig(mc=100.0, m1=1.0, m2=1.0, m3=1.0,
                          L1=0.5, L2=0.5, L3=0.5)
        from control.lqr import compute_lqr_gains
        K, A, B, P, Q, R = compute_lqr_gains(cfg)
        from control.closed_loop import compute_closed_loop
        cl = compute_closed_loop(A, B, K)
        assert cl["is_stable"]

    def test_light_links(self):
        """Very light links should still work."""
        cfg = SystemConfig(mc=2.0, m1=0.01, m2=0.01, m3=0.01,
                          L1=0.5, L2=0.5, L3=0.5)
        from control.lqr import compute_lqr_gains
        K, _, _, _, _, _ = compute_lqr_gains(cfg)
        assert K.shape == (1, 8)

    def test_unequal_lengths(self):
        """Very different link lengths."""
        cfg = SystemConfig(mc=2.0, m1=1.0, m2=1.0, m3=1.0,
                          L1=0.1, L2=0.5, L3=2.0)
        from control.lqr import compute_lqr_gains
        K, A, B, P, Q, R = compute_lqr_gains(cfg)
        from control.closed_loop import compute_closed_loop
        cl = compute_closed_loop(A, B, K)
        assert cl["is_stable"]
