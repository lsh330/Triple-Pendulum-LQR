"""Tests for all 8 equilibrium configurations."""
import pytest
import numpy as np
from parameters.equilibrium import (
    EQUILIBRIUM_CONFIGS, equilibrium, all_equilibria,
    equilibrium_potential_energy, all_potential_energies,
    config_index, config_name_from_index,
)
from dynamics.gravity.gravity_vector import gravity_vector


class TestEquilibriumDefinitions:
    """Verify equilibrium point definitions."""

    def test_eight_equilibria_exist(self):
        assert len(EQUILIBRIUM_CONFIGS) == 8

    def test_default_is_uuu(self):
        q = equilibrium()
        np.testing.assert_array_almost_equal(q, [0, np.pi, 0, 0])

    def test_backward_compatibility(self):
        """equilibrium() with no args returns original [0, pi, 0, 0]."""
        q = equilibrium("UUU")
        np.testing.assert_array_almost_equal(q, [0, np.pi, 0, 0])

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_gravity_zero_at_equilibrium(self, name, p):
        """G(q_eq) must be zero at every equilibrium."""
        q_eq = equilibrium(name)
        G = gravity_vector(q_eq, p)
        np.testing.assert_allclose(G, 0.0, atol=1e-12,
            err_msg=f"G(q_eq) != 0 at {name}")

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_absolute_angles_are_multiples_of_pi(self, name):
        """All absolute angles phi must be 0 or pi."""
        q = equilibrium(name)
        phi1 = q[1]
        phi2 = q[1] + q[2]
        phi3 = q[1] + q[2] + q[3]
        for phi in [phi1, phi2, phi3]:
            assert abs(np.sin(phi)) < 1e-12, \
                f"{name}: phi={phi}, sin(phi)={np.sin(phi)}"


class TestEquilibriumEnergy:
    """Verify potential energy ordering."""

    def test_ddd_lowest_energy(self, cfg):
        energies = all_potential_energies(cfg)
        assert energies["DDD"] == min(energies.values())

    def test_uuu_highest_energy(self, cfg):
        energies = all_potential_energies(cfg)
        assert energies["UUU"] == max(energies.values())

    def test_energy_symmetry(self, cfg):
        """V(DDD) = -V(UUU) by symmetry."""
        V_ddd = equilibrium_potential_energy(cfg, "DDD")
        V_uuu = equilibrium_potential_energy(cfg, "UUU")
        np.testing.assert_allclose(V_ddd, -V_uuu, rtol=1e-10)

    def test_energy_ordering(self, cfg):
        """Energy must increase: DDD < DDU < DUD < ... < UUU."""
        energies = all_potential_energies(cfg)
        values = [energies[name] for name in
                  ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"]]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], \
                f"Energy not increasing: {values[i]:.4f} >= {values[i+1]:.4f}"


class TestEquilibriumIndexing:
    def test_index_roundtrip(self):
        for i, name in enumerate(EQUILIBRIUM_CONFIGS.keys()):
            assert config_index(name) == i
            assert config_name_from_index(i) == name

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            equilibrium("INVALID")
