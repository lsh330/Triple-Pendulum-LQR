import numpy as np

from parameters.physical import PhysicalParams
from parameters.derived import compute_derived
from parameters.packing import pack as pack_derived
from parameters.equilibrium import equilibrium as _equilibrium


class SystemConfig:
    def __init__(self, mc: float, m1: float, m2: float, m3: float,
                 L1: float, L2: float, L3: float, g: float = 9.81):
        self._phys = PhysicalParams(mc=mc, m1=m1, m2=m2, m3=m3,
                                    L1=L1, L2=L2, L3=L3, g=g)
        self._derived = compute_derived(self._phys)

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
        return _equilibrium()
