"""Tests for multi-equilibrium LQR computation."""
import pytest
import numpy as np
from control.lqr import compute_all_lqr_gains
from parameters.equilibrium import EQUILIBRIUM_CONFIGS


class TestMultiEquilibriumLQR:
    @pytest.fixture(scope="class")
    def all_gains(self, cfg):
        return compute_all_lqr_gains(cfg)

    def test_all_eight_computed(self, all_gains):
        assert len(all_gains) == 8

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_each_equilibrium_has_result(self, all_gains, name):
        assert name in all_gains
        assert all_gains[name] is not None

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_controllability(self, all_gains, name):
        data = all_gains[name]
        if data is not None:
            assert data['is_controllable'] is True

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_closed_loop_stability(self, all_gains, name):
        data = all_gains[name]
        if data is not None:
            assert data['is_stable'] is True

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_gain_shape(self, all_gains, name):
        data = all_gains[name]
        if data is not None:
            K = data['K']
            assert K.shape == (1, 8)

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_riccati_positive_definite(self, all_gains, name):
        data = all_gains[name]
        if data is not None:
            P = data['P']
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals > 0), \
                f"P not PD at {name}: min eig = {eigvals.min()}"

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_closed_loop_poles_in_lhp(self, all_gains, name):
        """All closed-loop poles must have negative real part."""
        data = all_gains[name]
        if data is not None:
            poles = data['poles']
            max_real = np.max(np.real(poles))
            assert max_real < 0, \
                f"Unstable pole at {name}: max Re = {max_real}"

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_riccati_equation_satisfied(self, all_gains, name):
        """P must satisfy A'P + PA - PBR^{-1}B'P + Q = 0."""
        data = all_gains[name]
        if data is not None:
            A, B, P, Q, R = (data['A'], data['B'], data['P'],
                             data['Q'], data['R'])
            residual = (A.T @ P + P @ A
                        - P @ B @ np.linalg.solve(R, B.T @ P) + Q)
            np.testing.assert_allclose(residual, 0.0, atol=1e-6,
                err_msg=f"Riccati residual too large at {name}")

    @pytest.mark.parametrize("name", list(EQUILIBRIUM_CONFIGS.keys()))
    def test_equilibrium_position_correct(self, all_gains, name):
        """q_eq stored in results must match equilibrium() function."""
        from parameters.equilibrium import equilibrium
        data = all_gains[name]
        if data is not None:
            np.testing.assert_allclose(data['q_eq'], equilibrium(name),
                atol=1e-14, err_msg=f"q_eq mismatch at {name}")
