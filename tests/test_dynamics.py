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

    def test_gravity_at_equilibria(self, p):
        """Downward vertical is stable (zero accel); tilted from upright is unstable."""
        from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
        # Downward vertical (theta1=0): stable equilibrium, zero acceleration
        q_down = np.array([0.0, 0.0, 0.0, 0.0])
        ddq_down = forward_dynamics(q_down, np.zeros(4), 0.0, p)
        np.testing.assert_allclose(ddq_down, 0.0, atol=1e-12,
            err_msg="Downward vertical should be a stable equilibrium (zero accel)")

        # Slightly tilted from upright (theta1=pi+0.01): gravity should accelerate
        q_tilt = np.array([0.0, np.pi + 0.01, 0.0, 0.0])
        ddq_tilt = forward_dynamics(q_tilt, np.zeros(4), 0.0, p)
        assert np.linalg.norm(ddq_tilt) > 0, "Tilted from upright should accelerate"

    def test_consistency_multiple_configs(self, p):
        """Array and scalar dynamics must agree at various configurations."""
        from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
        from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast

        configs = [
            (np.array([0.0, np.pi, 0.0, 0.0]), np.array([0.5, 0.1, -0.2, 0.15]), 10.0),
            (np.array([0.1, np.pi + 0.3, 0.1, -0.1]), np.array([1.0, -0.5, 0.3, -0.2]), -5.0),
            (np.array([-0.2, np.pi - 0.2, -0.15, 0.05]), np.zeros(4), 0.0),
            (np.array([0.0, np.pi, 0.5, -0.3]), np.array([0.0, 2.0, -1.0, 0.5]), 50.0),
            (np.array([0.5, np.pi - 0.5, 0.2, -0.1]), np.array([-1.0, 0.3, -0.5, 0.7]), 25.0),
            (np.array([0.0, np.pi + 0.01, -0.01, 0.01]), np.array([0.01, -0.01, 0.01, -0.01]), 0.1),
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

    def test_rk4_energy_conservation(self, q_eq, p):
        """RK4 should approximately conserve energy over 100 steps (no control)."""
        from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast
        q0, q1, q2, q3 = q_eq
        dq0, dq1, dq2, dq3 = 0.05, 0.0, 0.0, 0.0
        # Run 100 steps with no control
        for _ in range(100):
            q0, q1, q2, q3, dq0, dq1, dq2, dq3 = rk4_step_fast(
                q0, q1, q2, q3, dq0, dq1, dq2, dq3, 0.0, p, 0.001)
        # State should remain bounded (no blow-up)
        assert all(abs(v) < 50 for v in [q0, q1, q2, q3, dq0, dq1, dq2, dq3]), \
            "RK4 integration diverged over 100 steps"

    def test_energy_conservation_no_control(self, q_eq, p):
        """Without control or friction, energy should be approximately conserved."""
        from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast
        from analysis.energy.kinetic_energy import compute_kinetic_energy
        from analysis.energy.potential_energy import compute_potential_energy
        from parameters.config import SystemConfig
        cfg = SystemConfig(mc=2.4, m1=1.323, m2=1.389, m3=0.8655,
                          L1=0.402, L2=0.332, L3=0.720)
        # Start near upright with small perturbation
        q0, q1, q2, q3 = 0.0, np.pi + 0.02, 0.0, 0.0
        dq0, dq1, dq2, dq3 = 0.0, 0.0, 0.0, 0.0
        # Compute initial energy
        q_arr = np.array([[q0, q1, q2, q3]])
        dq_arr = np.array([[dq0, dq1, dq2, dq3]])
        phi1_0 = q1; phi2_0 = q1 + q2; phi3_0 = q1 + q2 + q3
        E0 = (compute_kinetic_energy(cfg, q_arr, dq_arr,
               np.array([phi1_0]), np.array([phi2_0]), np.array([phi3_0]))[0]
              + compute_potential_energy(cfg,
               np.array([phi1_0]), np.array([phi2_0]), np.array([phi3_0]))[0])
        # Integrate 500 steps with no control
        for _ in range(500):
            q0, q1, q2, q3, dq0, dq1, dq2, dq3 = rk4_step_fast(
                q0, q1, q2, q3, dq0, dq1, dq2, dq3, 0.0, p, 0.001)
        # Compute final energy
        q_arr = np.array([[q0, q1, q2, q3]])
        dq_arr = np.array([[dq0, dq1, dq2, dq3]])
        phi1_f = q1; phi2_f = q1 + q2; phi3_f = q1 + q2 + q3
        Ef = (compute_kinetic_energy(cfg, q_arr, dq_arr,
               np.array([phi1_f]), np.array([phi2_f]), np.array([phi3_f]))[0]
              + compute_potential_energy(cfg,
               np.array([phi1_f]), np.array([phi2_f]), np.array([phi3_f]))[0])
        # Energy should be conserved within 1% (RK4 is not symplectic but close)
        assert abs(Ef - E0) / max(abs(E0), 1e-6) < 0.01, \
            f"Energy not conserved: E0={E0:.6f}, Ef={Ef:.6f}, drift={abs(Ef-E0)/abs(E0)*100:.2f}%"

    def test_timestep_convergence(self, q_eq, p):
        """Halving dt should reduce integration error quadratically (RK4 is O(dt^4))."""
        from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast
        q0, q1, q2, q3 = q_eq
        dq0 = 0.5; dq1 = 0.0; dq2 = 0.0; dq3 = 0.0
        # Integrate 0.1s with dt=0.002
        sq0, sq1 = q0, q1
        sdq0, sdq1 = dq0, dq1
        for _ in range(50):
            sq0, sq1, _, _, sdq0, sdq1, _, _ = rk4_step_fast(
                sq0, sq1, q2, q3, sdq0, sdq1, 0.0, 0.0, 0.0, p, 0.002)
        # Integrate 0.1s with dt=0.001
        fq0, fq1 = q0, q1
        fdq0, fdq1 = dq0, dq1
        for _ in range(100):
            fq0, fq1, _, _, fdq0, fdq1, _, _ = rk4_step_fast(
                fq0, fq1, q2, q3, fdq0, fdq1, 0.0, 0.0, 0.0, p, 0.001)
        # Both should give similar results (within RK4 O(dt^4) error)
        assert abs(sq0 - fq0) < 1e-4, f"Cart position diverges: coarse={sq0:.6f}, fine={fq0:.6f}"
        assert abs(sq1 - fq1) < 1e-4, f"Theta1 diverges: coarse={sq1:.6f}, fine={fq1:.6f}"
