"""Serialize derived parameters into a flat array for Numba JIT functions.

Packing order (13 elements):
    p[0]  = Mt   — total system mass (mc + m1 + m2 + m3)
    p[1]  = gx1  — cart-link1 coupling coefficient
    p[2]  = gx2  — cart-link2 coupling coefficient
    p[3]  = gx3  — cart-link3 coupling coefficient
    p[4]  = a1   — link1 rotational inertia coefficient
    p[5]  = a2   — link2 rotational inertia coefficient
    p[6]  = a3   — link3 rotational inertia coefficient
    p[7]  = b1   — link1-link2 coupling coefficient
    p[8]  = b2   — link1-link3 coupling coefficient
    p[9]  = b3   — link2-link3 coupling coefficient
    p[10] = gg1  — link1 gravity constant (gx1 * g)
    p[11] = gg2  — link2 gravity constant (gx2 * g)
    p[12] = gg3  — link3 gravity constant (gx3 * g)
"""

import numpy as np

# Parameter indices for documentation and safe access
IDX_MT = 0
IDX_GX1, IDX_GX2, IDX_GX3 = 1, 2, 3
IDX_A1, IDX_A2, IDX_A3 = 4, 5, 6
IDX_B1, IDX_B2, IDX_B3 = 7, 8, 9
IDX_GG1, IDX_GG2, IDX_GG3 = 10, 11, 12
N_PARAMS = 13

_KEYS = [
    "Mt", "gx1", "gx2", "gx3",
    "a1", "a2", "a3",
    "b1", "b2", "b3",
    "gg1", "gg2", "gg3",
]


def pack(derived: dict) -> np.ndarray:
    """Pack derived parameters into a flat float64 array of length 13."""
    return np.array([derived[k] for k in _KEYS], dtype=np.float64)
