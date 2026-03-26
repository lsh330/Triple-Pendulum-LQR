"""Tests for linearization: analytical vs numerical Jacobians, state-space properties."""

import numpy as np
import pytest
from parameters.config import SystemConfig


@pytest.fixture
def cfg():
    return SystemConfig(mc=2.4, m1=1.323, m2=1.389, m3=0.8655,
                        L1=0.402, L2=0.332, L3=0.720)


@pytest.fixture
def p(cfg):
    return cfg.pack()


@pytest.fixture
def q_eq(cfg):
    return cfg.equilibrium


class TestAnalyticalVsNumerical:
    def test_A_matrices_agree(self, q_eq, p):
        """Analytical and numerical Jacobians should produce similar A matrices."""
        from control.linearization.analytical_jacobian import compute_analytical_jacobians
        from control.linearization.jit_jacobians import compute_jacobians_jit

        A_ana, B_ana = compute_analytical_jacobians(q_eq, p)
        A_num, B_num = compute_jacobians_jit(q_eq, np.zeros(4), 0.0, p)

        np.testing.assert_allclose(A_ana, A_num, atol=1e-4,
                                   err_msg="Analytical and numerical A should agree")

    def test_B_matrices_agree(self, q_eq, p):
        """Analytical and numerical B should agree."""
        from control.linearization.analytical_jacobian import compute_analytical_jacobians
        from control.linearization.jit_jacobians import compute_jacobians_jit

        A_ana, B_ana = compute_analytical_jacobians(q_eq, p)
        A_num, B_num = compute_jacobians_jit(q_eq, np.zeros(4), 0.0, p)

        np.testing.assert_allclose(B_ana, B_num, atol=1e-4,
                                   err_msg="Analytical and numerical B should agree")


class TestStateSpace:
    def test_A_shape(self, q_eq, p):
        from control.linearization.linearize import linearize
        A, B = linearize(q_eq, p)
        assert A.shape == (8, 8)

    def test_B_shape(self, q_eq, p):
        from control.linearization.linearize import linearize
        A, B = linearize(q_eq, p)
        assert B.shape == (8, 1)

    def test_top_right_identity(self, q_eq, p):
        """A should have identity in top-right block (velocity -> position)."""
        from control.linearization.linearize import linearize
        A, B = linearize(q_eq, p)
        np.testing.assert_allclose(A[:4, 4:], np.eye(4), atol=1e-10)

    def test_top_left_zero(self, q_eq, p):
        """A should have zeros in top-left block."""
        from control.linearization.linearize import linearize
        A, B = linearize(q_eq, p)
        np.testing.assert_allclose(A[:4, :4], 0.0, atol=1e-10)

    def test_open_loop_unstable(self, q_eq, p):
        """Triple inverted pendulum should be open-loop unstable."""
        from control.linearization.linearize import linearize
        A, B = linearize(q_eq, p)
        eigs = np.linalg.eigvals(A)
        n_unstable = np.sum(eigs.real > 0)
        assert n_unstable >= 1, "System should have at least 1 unstable mode"

    def test_controllable(self, q_eq, p):
        """System should be controllable (rank of controllability matrix = n)."""
        from control.linearization.linearize import linearize
        A, B = linearize(q_eq, p)
        n = 8
        C_mat = B.copy()
        AB = B.copy()
        for i in range(1, n):
            AB = A @ AB
            C_mat = np.hstack([C_mat, AB])
        rank = np.linalg.matrix_rank(C_mat)
        assert rank == n, f"System not controllable, rank={rank}"
