import numpy as np


def apply_lowpass(white, dt, bandwidth):
    """Apply a lowpass filter using FFT with Butterworth rolloff.

    H(f) = 1 / (1 + (f/bandwidth)^4)
    """
    N = len(white)
    freqs = np.fft.rfftfreq(N, d=dt)
    spectrum = np.fft.rfft(white)

    # Butterworth rolloff (avoid division by zero at f=0)
    H = 1.0 / (1.0 + (freqs / bandwidth) ** 4)
    spectrum *= H

    filtered = np.fft.irfft(spectrum, n=N)
    return filtered
