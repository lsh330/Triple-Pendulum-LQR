"""Compose all frequency-domain analysis sub-modules into one dict."""

import numpy as np

from analysis.frequency.open_loop_response import compute_open_loop_response
from analysis.frequency.closed_loop_response import compute_closed_loop_response
from analysis.frequency.sensitivity import compute_sensitivity
from analysis.frequency.stability_margins import compute_margins
from analysis.frequency.poles import compute_poles
from analysis.frequency.step_response import compute_step_response


def compute_frequency_response(A, B, K, A_cl, w=None) -> dict:
    """Return a combined dict with all frequency-domain analysis results.

    Parameters
    ----------
    A : (n, n) open-loop system matrix
    B : (n,) or (n, 1) input matrix
    K : (n,) or (1, n) LQR gain vector
    A_cl : (n, n) closed-loop system matrix  (A - B @ K)
    w : 1-D array of angular frequencies (rad/s), optional
    """
    if w is None:
        w = np.logspace(-2, 3, 2000)

    B = np.asarray(B).ravel()
    K = np.asarray(K).ravel()

    ol = compute_open_loop_response(A, B, K, w)
    cl = compute_closed_loop_response(A_cl, B, w)
    sens = compute_sensitivity(ol["L_jw"])
    margins = compute_margins(ol["w"], ol["L_mag"], ol["L_phase_deg"])
    pole_data = compute_poles(A, A_cl)
    step_data = compute_step_response(cl["sys_CL"])

    return {**ol, **cl, **sens, **margins, **pole_data, **step_data}
