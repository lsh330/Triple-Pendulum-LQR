"""Tests for dynamics computations: mass matrix, gravity, Coriolis, forward dynamics."""

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


class TestMassMatrix:
    def test_symmetric(self, q_eq, p):
        from dynamics.mass_matrix.assembly import mass_matrix
        M = mass_matrix(q_eq, p)
        np.testing.assert_allclose(M, M.T, atol=1e-14)

    def test_positive_definite(self, q_eq, p):
        from dynamics.mass_matrix.assembly import mass_matrix
        M = mass_matrix(q_eq, p)
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs > 0), f"Mass matrix not positive definite: {eigs}"

    def test_total_mass(self, cfg, q_eq, p):
        from dynamics.mass_matrix.assembly import mass_matrix
        M = mass_matrix(q_eq, p)
        expected_Mt = cfg.mc + cfg.m1 + cfg.m2 + cfg.m3
        assert abs(M[0, 0] - expected_Mt) < 1e-12

    def test_varies_with_config(self, p):
        from dynamics.mass_matrix.assembly import mass_matrix
        q1 = np.array([0.0, np.pi, 0.0, 0.0])
        q2 = np.array([0.0, np.pi + 0.1, 0.0, 0.0])
        M1 = mass_matrix(q1, p)
        M2 = mass_matrix(q2, p)
        assert not np.allclose(M1, M2), "Mass matrix should depend on configuration"


class TestGravityVector:
    def test_cart_row_zero(self, q_eq, p):
        from dynamics.gravity.gravity_vector import gravity_vector
        G = gravity_vector(q_eq, p)
        assert abs(G[0]) < 1e-14, "Gravity should not act on cart DOF"

    def test_nonzero_at_eq(self, q_eq, p):
        from dynamics.gravity.gravity_vector import gravity_vector
        G = gravity_vector(q_eq, p)
        # At upright equilibrium (theta1=pi), gravity terms are nonzero
        assert np.linalg.norm(G[1:]) > 0


class TestCoriolisVector:
    def test_zero_at_rest(self, q_eq, p):
        from dynamics.coriolis.christoffel import coriolis_vector
        dq = np.zeros(4)
        C = coriolis_vector(q_eq, dq, p)
        np.testing.assert_allclose(C, 0.0, atol=1e-14)

    def test_nonzero_with_velocity(self, q_eq, p):
        from dynamics.coriolis.christoffel import coriolis_vector
        dq = np.array([1.0, 0.5, 0.3, 0.2])
        C = coriolis_vector(q_eq, dq, p)
        assert np.linalg.norm(C) > 0


class TestForwardDynamics:
    def test_consistency(self, q_eq, p):
        """Array and scalar forward dynamics should agree."""
        from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
        from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast

        dq = np.array([0.5, 0.1, -0.2, 0.15])
        u = 10.0

        ddq_arr = forward_dynamics(q_eq, dq, u, p)
        ddq0, ddq1, ddq2, ddq3 = forward_dynamics_fast(
            q_eq[0], q_eq[1], q_eq[2], q_eq[3],
            dq[0], dq[1], dq[2], dq[3], u, p)
        ddq_scalar = np.array([ddq0, ddq1, ddq2, ddq3])

        np.testing.assert_allclose(ddq_arr, ddq_scalar, atol=1e-10)

    def test_free_fall(self, p):
        """At downward vertical (theta1=0), with no control, gravity should accelerate."""
        from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
        q = np.array([0.0, 0.0, 0.0, 0.0])
        dq = np.zeros(4)
        ddq = forward_dynamics(q, dq, 0.0, p)
        # System should accelerate under gravity
        assert np.linalg.norm(ddq) > 0

    def test_consistency_multiple_configs(self, p):
        """Array and scalar dynamics must agree at various configurations."""
        from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
        from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast

        configs = [
            (np.array([0.0, np.pi, 0.0, 0.0]), np.array([0.5, 0.1, -0.2, 0.15]), 10.0),
            (np.array([0.1, np.pi + 0.3, 0.1, -0.1]), np.array([1.0, -0.5, 0.3, -0.2]), -5.0),
            (np.array([-0.2, np.pi - 0.2, -0.15, 0.05]), np.zeros(4), 0.0),
            (np.array([0.0, np.pi, 0.5, -0.3]), np.array([0.0, 2.0, -1.0, 0.5]), 50.0),
        ]
        for q, dq, u in configs:
            ddq_arr = forward_dynamics(q, dq, u, p)
            ddq0, ddq1, ddq2, ddq3 = forward_dynamics_fast(
                q[0], q[1], q[2], q[3], dq[0], dq[1], dq[2], dq[3], u, p)
            ddq_scalar = np.array([ddq0, ddq1, ddq2, ddq3])
            np.testing.assert_allclose(ddq_arr, ddq_scalar, atol=1e-10,
                err_msg=f"Mismatch at q={q}, dq={dq}, u={u}")

    def test_rk4_energy_bounded(self, q_eq, p):
        """RK4 step should not blow up energy for small dt."""
        from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast
        q0, q1, q2, q3 = q_eq
        dq0, dq1, dq2, dq3 = 0.1, 0.0, 0.0, 0.0
        nq0, nq1, nq2, nq3, nd0, nd1, nd2, nd3 = rk4_step_fast(
            q0, q1, q2, q3, dq0, dq1, dq2, dq3, 0.0, p, 0.001)
        # State should not explode
        assert all(abs(v) < 100 for v in [nq0, nq1, nq2, nq3, nd0, nd1, nd2, nd3])
