import numpy as np


def normalize_rms(signal, target_amplitude):
    """Scale signal so that its RMS equals target_amplitude."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-15:
        return signal
    return signal * (target_amplitude / rms)
