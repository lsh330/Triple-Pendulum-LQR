"""Validation tests for energy conservation."""
import pytest
import numpy as np
from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast
from control.swing_up.energy_computation import total_energy_scalar


class TestEnergyConservation:
    """Verify energy conservation in the uncontrolled system."""

    def test_free_swing_energy_conserved(self, p):
        """With u=0, total energy must be conserved (< 0.01% drift per second)."""
        # Start near DDD with small perturbation
        sq0, sq1, sq2, sq3 = 0.0, 0.1, 0.0, 0.0
        sdq0, sdq1, sdq2, sdq3 = 0.0, 0.0, 0.0, 0.0
        dt = 0.001

        E0 = total_energy_scalar(sq0, sq1, sq2, sq3,
                                 sdq0, sdq1, sdq2, sdq3, p)

        for _ in range(1000):  # 1 second
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
                sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, 0.0, p, dt)

        E_final = total_energy_scalar(sq0, sq1, sq2, sq3,
                                      sdq0, sdq1, sdq2, sdq3, p)
        drift = abs(E_final - E0) / (abs(E0) + 1e-10)
        assert drift < 0.0001, f"Energy drift = {drift * 100:.4f}%"

    def test_energy_conserved_large_oscillation(self, p):
        """Energy conservation during large-amplitude oscillation."""
        sq0, sq1, sq2, sq3 = 0.0, 1.5, 0.5, -0.3
        sdq0, sdq1, sdq2, sdq3 = 0.0, 2.0, -1.0, 0.5
        dt = 0.0005  # Smaller step for large oscillation

        E0 = total_energy_scalar(sq0, sq1, sq2, sq3,
                                 sdq0, sdq1, sdq2, sdq3, p)

        for _ in range(2000):  # 1 second
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
                sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, 0.0, p, dt)

        E_final = total_energy_scalar(sq0, sq1, sq2, sq3,
                                      sdq0, sdq1, sdq2, sdq3, p)
        drift = abs(E_final - E0) / (abs(E0) + 1e-10)
        assert drift < 0.001, f"Energy drift = {drift * 100:.4f}%"


class TestCoriolisSkewSymmetry:
    """Verify M_dot - 2C is skew-symmetric (passivity property).

    C(q,dq) is defined via Christoffel symbols:
      C_ij(q,dq) = sum_k Gamma_ijk(q) * dq_k
      Gamma_ijk   = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)

    The passivity property states that M_dot(q,dq) - 2*C(q,dq)
    is skew-symmetric, i.e., z^T (M_dot - 2C) z = 0 for all z.
    """

    def _build_coriolis_matrix(self, q, dq, p):
        """Build C(q,dq) matrix from numerical mass-matrix partial derivatives."""
        from dynamics.mass_matrix.assembly import mass_matrix
        n = 4
        eps = 1e-7
        # dM[i,j,k] = dM_ij / dq_k  (central difference)
        dM = np.zeros((n, n, n))
        for k in range(n):
            ek = np.zeros(n)
            ek[k] = 1.0
            dM[:, :, k] = (mass_matrix(q + eps * ek, p)
                           - mass_matrix(q - eps * ek, p)) / (2 * eps)
        # C_ij = sum_k Gamma_ijk * dq_k
        C_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Gamma = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                    C_mat[i, j] += Gamma * dq[k]
        return C_mat

    def test_coriolis_matrix_matches_vector(self, p):
        """C_mat(q,dq) @ dq must equal coriolis_vector(q, dq, p)."""
        from dynamics.coriolis.christoffel import coriolis_vector
        rng = np.random.default_rng(77)
        q = rng.uniform(-np.pi, np.pi, 4)
        dq = rng.uniform(-3, 3, 4)
        C_mat = self._build_coriolis_matrix(q, dq, p)
        h_mat = C_mat @ dq
        h_vec = coriolis_vector(q, dq, p)
        np.testing.assert_allclose(h_mat, h_vec, atol=1e-6,
            err_msg="Christoffel C_mat @ dq != coriolis_vector")

    def test_skew_symmetry(self, p):
        """z^T (M_dot - 2C) z must be zero for arbitrary z, q, dq."""
        from dynamics.mass_matrix.assembly import mass_matrix

        rng = np.random.default_rng(123)
        eps = 1e-7
        for _ in range(10):
            q = rng.uniform(-np.pi, np.pi, 4)
            dq = rng.uniform(-5, 5, 4)
            z = rng.uniform(-1, 1, 4)

            # M_dot = sum_k (dM/dq_k) * dq_k  (directional derivative along dq)
            M_dot = np.zeros((4, 4))
            for k in range(4):
                ek = np.zeros(4)
                ek[k] = 1.0
                M_pk = mass_matrix(q + eps * ek, p)
                M_mk = mass_matrix(q - eps * ek, p)
                M_dot += (M_pk - M_mk) / (2 * eps) * dq[k]

            C_mat = self._build_coriolis_matrix(q, dq, p)

            # Skew-symmetry test: z^T (M_dot - 2C) z should be ~0
            S = M_dot - 2 * C_mat
            result = z @ S @ z
            assert abs(result) < 1e-5, \
                f"Skew-symmetry violated: z^T(Mdot-2C)z = {result}"
