import numpy as np


def pack(derived: dict) -> np.ndarray:
    keys = [
        "Mt", "gx1", "gx2", "gx3",
        "a1", "a2", "a3",
        "b1", "b2", "b3",
        "gg1", "gg2", "gg3",
    ]
    return np.array([derived[k] for k in keys], dtype=np.float64)
