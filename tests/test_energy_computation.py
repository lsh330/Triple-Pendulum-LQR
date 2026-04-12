"""Tests for energy computation functions."""
import pytest
import numpy as np
from control.swing_up.energy_computation import total_energy_scalar, target_energy_from_phis
from parameters.equilibrium import equilibrium, EQUILIBRIUM_CONFIGS


class TestTotalEnergy:
    def test_zero_at_rest_ddd(self, p):
        """At DDD equilibrium with zero velocity, E = V_physical(DDD)."""
        q = equilibrium("DDD")
        E = total_energy_scalar(q[0], q[1], q[2], q[3], 0, 0, 0, 0, p)
        E_target = target_energy_from_phis(0.0, 0.0, 0.0, p)
        np.testing.assert_allclose(E, E_target, atol=1e-10)

    def test_energy_matches_target_at_each_equilibrium(self, p):
        """At each equilibrium (zero velocity), E must equal E*."""
        for name, (phi1, phi2, phi3) in EQUILIBRIUM_CONFIGS.items():
            q = equilibrium(name)
            E = total_energy_scalar(q[0], q[1], q[2], q[3], 0, 0, 0, 0, p)
            E_target = target_energy_from_phis(phi1, phi2, phi3, p)
            np.testing.assert_allclose(E, E_target, atol=1e-10,
                err_msg=f"Energy mismatch at {name}")

    def test_kinetic_energy_positive(self, p):
        """Energy with velocity must be higher than at rest."""
        q = equilibrium("DDD")
        E_rest = total_energy_scalar(q[0], q[1], q[2], q[3], 0, 0, 0, 0, p)
        E_moving = total_energy_scalar(q[0], q[1], q[2], q[3], 1, 0, 0, 0, p)
        assert E_moving > E_rest

    def test_energy_symmetric(self, p):
        """E(DDD, zero vel) = -E(UUU, zero vel)."""
        q_ddd = equilibrium("DDD")
        q_uuu = equilibrium("UUU")
        E_ddd = total_energy_scalar(
            q_ddd[0], q_ddd[1], q_ddd[2], q_ddd[3], 0, 0, 0, 0, p)
        E_uuu = total_energy_scalar(
            q_uuu[0], q_uuu[1], q_uuu[2], q_uuu[3], 0, 0, 0, 0, p)
        np.testing.assert_allclose(E_ddd, -E_uuu, rtol=1e-10)


class TestSwingUpControl:
    def test_zero_at_target_energy(self, p):
        """Swing-up control must be zero when E = E_target.

        Constructs a state at DDD position with dx chosen so that
        total KE + PE_DDD == E_target(UUU), then verifies u = k*(0)*dx = 0.
        """
        from control.swing_up.energy_controller import energy_swing_up_control
        E_target = target_energy_from_phis(np.pi, np.pi, np.pi, p)
        # V_physical at DDD = -E_target (by symmetry); solve for dx:
        # 0.5*Mt*dx^2 + V_DDD = E_target  =>  dx^2 = 2*(E_target - V_DDD) / Mt
        Mt = p[0]
        V_ddd = target_energy_from_phis(0.0, 0.0, 0.0, p)
        dx = np.sqrt(2.0 * (E_target - V_ddd) / Mt)
        # Verify E == E_target
        E_check = total_energy_scalar(0.0, 0.0, 0.0, 0.0, dx, 0.0, 0.0, 0.0, p)
        np.testing.assert_allclose(E_check, E_target, atol=1e-8)
        # Control should be zero: u = k*(E_target - E_target)*dx = 0
        u = energy_swing_up_control(0.0, 0.0, 0.0, 0.0, dx, 0.0, 0.0, 0.0,
                                    p, E_target, 50.0, 200.0)
        np.testing.assert_allclose(u, 0.0, atol=1e-8)

    def test_zero_when_cart_stationary(self, p):
        """Swing-up control must be zero when dx = 0."""
        from control.swing_up.energy_controller import energy_swing_up_control
        q = equilibrium("DDD")
        E_target = target_energy_from_phis(np.pi, np.pi, np.pi, p)
        u = energy_swing_up_control(q[0], q[1], q[2], q[3], 0, 0, 0, 0,
                                    p, E_target, 50.0, 200.0)
        assert u == 0.0

    def test_saturation(self, p):
        """Control must be clamped to u_max."""
        from control.swing_up.energy_controller import energy_swing_up_control
        q = equilibrium("DDD")
        E_target = target_energy_from_phis(np.pi, np.pi, np.pi, p)
        u = energy_swing_up_control(q[0], q[1], q[2], q[3], 10.0, 0, 0, 0,
                                    p, E_target, 1000.0, 50.0)
        assert abs(u) <= 50.0
