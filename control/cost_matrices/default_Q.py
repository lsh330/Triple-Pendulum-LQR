import numpy as np


def default_Q():
    """Return the default 8x8 state cost matrix."""
    return np.diag([10.0, 100.0, 100.0, 100.0, 1.0, 10.0, 10.0, 10.0])
