from dataclasses import dataclass


@dataclass
class PhysicalParams:
    mc: float
    m1: float
    m2: float
    m3: float
    L1: float
    L2: float
    L3: float
    g: float = 9.81
