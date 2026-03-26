"""Compute gain and phase margins from open-loop frequency response."""

import numpy as np


def compute_margins(w, L_mag, L_phase_deg) -> dict:
    """Return dict with phase_margin, wgc, gain_margin_dB, wpc.

    * Gain crossover: |L| crosses 1 (0 dB) from above.
    * Phase crossover: phase crosses -180 deg.
    """
    # --- Gain crossover (|L| = 1) ---
    log_mag = np.log10(L_mag + 1e-30)
    phase_margin = np.nan
    wgc = np.nan
    for i in range(len(log_mag) - 1):
        if log_mag[i] >= 0 and log_mag[i + 1] < 0:
            # Linear interpolation
            frac = log_mag[i] / (log_mag[i] - log_mag[i + 1])
            wgc = w[i] + frac * (w[i + 1] - w[i])
            phase_at_gc = L_phase_deg[i] + frac * (L_phase_deg[i + 1] - L_phase_deg[i])
            phase_margin = 180.0 + phase_at_gc
            break

    # --- Phase crossover (phase = -180 deg) ---
    gain_margin_dB = np.nan
    wpc = np.nan
    phase_shifted = L_phase_deg + 180.0
    for i in range(len(phase_shifted) - 1):
        if phase_shifted[i] >= 0 and phase_shifted[i + 1] < 0:
            frac = phase_shifted[i] / (phase_shifted[i] - phase_shifted[i + 1])
            wpc = w[i] + frac * (w[i + 1] - w[i])
            mag_at_pc = L_mag[i] + frac * (L_mag[i + 1] - L_mag[i])
            gain_margin_dB = -20.0 * np.log10(mag_at_pc + 1e-30)
            break

    return dict(phase_margin=phase_margin, wgc=wgc,
                gain_margin_dB=gain_margin_dB, wpc=wpc)
