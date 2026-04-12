"""Form-switching supervisor for transitioning between equilibrium configurations.

The supervisor pre-computes:
  - LQR gains (K, P) for every one of the 8 equilibrium configurations
  - Lyapunov ROA thresholds (rho_in, rho_out) for each target config
  - Target mechanical energies for the energy-based swing-up law

All data can then be packed into flat numpy arrays via :meth:`pack_for_njit`
for use inside a zero-overhead ``@njit`` simulation loop.
"""

import numpy as np

from parameters.equilibrium import EQUILIBRIUM_CONFIGS, equilibrium_potential_energy
from control.supervisor.transition_graph import plan_transition
from control.supervisor.roa_estimation import estimate_lyapunov_roa


class FormSwitchSupervisor:
    """Manage transitions between equilibrium configurations.

    Uses energy-based swing-up combined with LQR catch, with hysteresis
    switching governed by Lyapunov level sets.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration.
    k_energy : float
        Energy shaping gain for the swing-up controller.
    u_max : float
        Actuator saturation limit (N).
    """

    def __init__(self, cfg, k_energy: float = 50.0, u_max: float = 200.0):
        self.cfg = cfg
        self.k_energy = k_energy
        self.u_max = u_max

        # Lazy import to avoid circular dependencies at module load time
        from control.lqr import compute_all_lqr_gains

        # Compute LQR gains for all 8 equilibria
        self._eq_data = compute_all_lqr_gains(cfg)

        # Estimate Lyapunov ROA for each stable equilibrium
        self._roa_data: dict = {}
        for name, data in self._eq_data.items():
            if data is not None and data["is_stable"]:
                roa = estimate_lyapunov_roa(
                    cfg,
                    data["K"],
                    data["P"],
                    data["q_eq"],
                    n_samples=300,
                    max_angle_deg=25.0,
                    t_horizon=3.0,
                )
                self._roa_data[name] = roa

        # Pre-compute target physical energies for swing-up
        self._target_energies: dict = {
            name: equilibrium_potential_energy(cfg, name)
            for name in EQUILIBRIUM_CONFIGS
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def plan(self, source: str, target: str) -> list:
        """Return a minimal-hop transition path from *source* to *target*.

        Parameters
        ----------
        source, target : str
            Configuration names.

        Returns
        -------
        list of str
            Path including both endpoints.
        """
        return plan_transition(source, target)

    def pack_for_njit(self, transition_path: list) -> dict:
        """Pack all supervisor data into flat numpy arrays for the JIT loop.

        Parameters
        ----------
        transition_path : list of str
            Ordered list of configuration names, e.g.
            ``["DDD", "UDD", "UUD", "UUU"]``.

        Returns
        -------
        dict
            Keys: ``n_stages``, ``all_q_eq`` (n,4), ``all_K_flat`` (n,8),
            ``all_P_flat`` (n,8,8), ``all_E_target`` (n,),
            ``all_rho_in`` (n,), ``all_rho_out`` (n,),
            ``k_energy``, ``u_max``.

        Raises
        ------
        ValueError
            If any target configuration in *transition_path* lacks LQR data.
        """
        n_stages = len(transition_path) - 1  # number of swing-up/catch stages

        all_q_eq = np.zeros((n_stages, 4))
        all_K_flat = np.zeros((n_stages, 8))
        all_P_flat = np.zeros((n_stages, 8, 8))
        all_E_target = np.zeros(n_stages)
        all_rho_in = np.zeros(n_stages)
        all_rho_out = np.zeros(n_stages)

        for i in range(n_stages):
            target_name = transition_path[i + 1]
            data = self._eq_data.get(target_name)
            if data is None:
                raise ValueError(
                    f"No valid LQR data for configuration '{target_name}'. "
                    "Cannot include this step in the switching path."
                )

            all_q_eq[i] = data["q_eq"]
            all_K_flat[i] = data["K"].flatten()
            all_P_flat[i] = data["P"]
            all_E_target[i] = self._target_energies[target_name]

            roa = self._roa_data.get(target_name)
            if roa is not None:
                all_rho_in[i] = roa["rho_in"]
                all_rho_out[i] = roa["rho_out"]
            else:
                # Conservative fall-back: small catch window, wide exit
                all_rho_in[i] = 1.0
                all_rho_out[i] = 2.0

        return {
            "n_stages": n_stages,
            "all_q_eq": all_q_eq,
            "all_K_flat": all_K_flat,
            "all_P_flat": all_P_flat,
            "all_E_target": all_E_target,
            "all_rho_in": all_rho_in,
            "all_rho_out": all_rho_out,
            "k_energy": float(self.k_energy),
            "u_max": float(self.u_max),
        }
