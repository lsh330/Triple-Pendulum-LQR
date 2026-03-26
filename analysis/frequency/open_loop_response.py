"""Compute the open-loop frequency response L(jw) = K (jwI - A)^{-1} B."""

import numpy as np
import scipy.signal as sig


def compute_open_loop_response(A, B, K, w) -> dict:
    """Return dict with w, L_jw, L_mag, L_phase_deg.

    Phase is computed without unwrap to avoid issues
    with unstable open-loop systems (inverted pendulum).
    """
    C_L = K.reshape(1, -1)
    D_L = np.zeros((1, 1))
    sys_L = sig.lti(A, B.reshape(-1, 1), C_L, D_L)

    w_out, H = sig.freqresp(sys_L, w=w)

    L_jw = H.flatten()
    L_mag = np.abs(L_jw)
    L_phase_deg = np.degrees(np.angle(L_jw))  # wrapped [-180, 180]

    return dict(w=w_out, L_jw=L_jw, L_mag=L_mag, L_phase_deg=L_phase_deg)
