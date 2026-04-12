"""Tests for core utilities."""
import pytest
import numpy as np
from core.state_index import QIndex, DQIndex, XIndex, PIndex
from core.angle_utils import angle_wrap


class TestStateIndex:
    def test_q_dimensions(self):
        assert QIndex.DIM == 4
        q = np.zeros(4)
        assert q[QIndex.CART] == 0.0
        assert q[QIndex.THETA1] == 0.0

    def test_x_dimensions(self):
        assert XIndex.DIM == 8
        x = np.arange(8, dtype=float)
        assert x[XIndex.CART_POS] == 0.0
        assert x[XIndex.OMEGA3] == 7.0

    def test_p_dimensions(self):
        assert PIndex.DIM == 13

    def test_q_angles_slice(self):
        q = np.array([0.0, 1.0, 2.0, 3.0])
        angles = q[QIndex.ANGLES]
        np.testing.assert_array_equal(angles, [1.0, 2.0, 3.0])

    def test_dq_angular_slice(self):
        dq = np.array([0.5, 1.0, 2.0, 3.0])
        angular = dq[DQIndex.ANGULAR]
        np.testing.assert_array_equal(angular, [1.0, 2.0, 3.0])

    def test_x_q_dq_slices(self):
        x = np.arange(8, dtype=float)
        np.testing.assert_array_equal(x[XIndex.Q], [0, 1, 2, 3])
        np.testing.assert_array_equal(x[XIndex.DQ], [4, 5, 6, 7])


class TestAngleWrap:
    def test_zero(self):
        assert angle_wrap(0.0) == 0.0

    def test_pi(self):
        # pi and -pi are equivalent; floor-based wrap returns -pi
        result = angle_wrap(np.pi)
        assert abs(abs(result) - np.pi) < 1e-12

    def test_minus_pi(self):
        np.testing.assert_allclose(angle_wrap(-np.pi), -np.pi, atol=1e-12)

    def test_wrap_positive(self):
        # 3*pi wraps to pi or -pi (equivalent)
        result = angle_wrap(3 * np.pi)
        assert abs(abs(result) - np.pi) < 1e-12

    def test_wrap_negative(self):
        np.testing.assert_allclose(angle_wrap(-3 * np.pi), -np.pi, atol=1e-12)

    def test_within_range(self):
        assert angle_wrap(1.0) == 1.0
        assert angle_wrap(-1.0) == -1.0

    def test_two_pi_wraps_to_zero(self):
        np.testing.assert_allclose(angle_wrap(2 * np.pi), 0.0, atol=1e-12)

    def test_output_in_range(self):
        """For any input, output must be in [-pi, pi]."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(-10 * np.pi, 10 * np.pi, 50)
        for a in angles:
            w = angle_wrap(a)
            assert -np.pi - 1e-12 <= w <= np.pi + 1e-12, \
                f"angle_wrap({a}) = {w} out of range"
