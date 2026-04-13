import numpy as np

from parameters.physical import PhysicalParams
from parameters.derived import compute_derived
from parameters.packing import pack as pack_derived
from parameters.equilibrium import (
    equilibrium as _equilibrium,
    all_equilibria as _all_equilibria,
    equilibrium_potential_energy as _eq_pe,
    EQUILIBRIUM_CONFIGS,
)


class SystemConfig:
    # 액추에이터 포화 기본값 (N) — W1: ROA 모듈 간 일관성을 위해 중앙 관리
    DEFAULT_ACTUATOR_SATURATION: float = 200.0

    def __init__(self, mc: float, m1: float, m2: float, m3: float,
                 L1: float, L2: float, L3: float, g: float = 9.81,
                 target_equilibrium: str = "UUU",
                 actuator_saturation: float = 200.0):
        # Validate that mass and length parameters are positive and g > 0
        for name, val in [('mc', mc), ('m1', m1), ('m2', m2), ('m3', m3),
                          ('L1', L1), ('L2', L2), ('L3', L3)]:
            if val <= 0:
                raise ValueError(f"Parameter '{name}' must be positive, got {val}")
        if g <= 0:
            raise ValueError(f"Parameter 'g' must be positive, got {g}")
        if actuator_saturation <= 0:
            raise ValueError(f"Parameter 'actuator_saturation' must be positive, got {actuator_saturation}")
        if target_equilibrium not in EQUILIBRIUM_CONFIGS:
            raise ValueError(
                f"Unknown target_equilibrium '{target_equilibrium}'. "
                f"Choose from: {list(EQUILIBRIUM_CONFIGS.keys())}"
            )

        self._phys = PhysicalParams(mc=mc, m1=m1, m2=m2, m3=m3,
                                    L1=L1, L2=L2, L3=L3, g=g)
        self._derived = compute_derived(self._phys)
        self._target_eq = target_equilibrium
        # W1: 액추에이터 포화 한계 — ROA/제어 모듈이 cfg에서 참조
        self.actuator_saturation: float = float(actuator_saturation)

    # Expose physical params as attributes
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._phys, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def pack(self) -> np.ndarray:
        return pack_derived(self._derived)

    @property
    def equilibrium(self) -> np.ndarray:
        """Return q_eq for the currently selected target equilibrium."""
        return _equilibrium(self._target_eq)

    def set_equilibrium(self, config_name: str) -> None:
        """Switch the target equilibrium configuration.

        Parameters
        ----------
        config_name : str
            One of "DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU".
        """
        if config_name not in EQUILIBRIUM_CONFIGS:
            raise ValueError(
                f"Unknown config '{config_name}'. "
                f"Choose from: {list(EQUILIBRIUM_CONFIGS.keys())}"
            )
        self._target_eq = config_name

    def all_equilibria(self) -> dict:
        """Return dict mapping config_name -> q_eq array for all 8 configurations."""
        return _all_equilibria()

    def equilibrium_energy(self, config_name: str) -> float:
        """Return physical potential energy V_physical (J) at the given equilibrium.

        Parameters
        ----------
        config_name : str
            Equilibrium configuration name.

        Returns
        -------
        float
            V_physical = -PE_code at the equilibrium (KE=0).
        """
        return _eq_pe(self, config_name)
