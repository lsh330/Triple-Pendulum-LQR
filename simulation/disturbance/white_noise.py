import numpy as np


def generate_white_noise(N, seed=42):
    """Generate white noise of length N using a seeded RandomState."""
    rng = np.random.RandomState(seed)
    return rng.randn(N)
