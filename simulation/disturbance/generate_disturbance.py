import numpy as np
from simulation.disturbance.white_noise import generate_white_noise
from simulation.disturbance.bandpass_filter import apply_lowpass
from simulation.disturbance.normalize import normalize_rms


def generate_disturbance(t_arr, amplitude=20.0, bandwidth=5.0, seed=42):
    """Generate a band-limited disturbance force signal.

    Parameters
    ----------
    t_arr : ndarray
        Time array.
    amplitude : float
        Target RMS amplitude of the disturbance.
    bandwidth : float
        Lowpass filter bandwidth in Hz.
    seed : int
        Random seed.

    Returns
    -------
    d_arr : ndarray
        Disturbance force array, same length as t_arr.
    """
    N = len(t_arr)
    dt = t_arr[1] - t_arr[0]

    nyquist_freq = 0.5 / dt
    if bandwidth > nyquist_freq:
        from utils.logger import get_logger
        get_logger().warning("Disturbance bandwidth (%.1f Hz) exceeds Nyquist frequency (%.1f Hz). "
                           "Aliasing will occur.", bandwidth, nyquist_freq)

    white = generate_white_noise(N, seed=seed)
    filtered = apply_lowpass(white, dt, bandwidth)
    d_arr = normalize_rms(filtered, amplitude)
    return d_arr
