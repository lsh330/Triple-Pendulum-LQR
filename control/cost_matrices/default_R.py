import numpy as np


def default_R():
    """Return the default 1x1 control cost matrix."""
    return np.array([[0.01]])
