"""Compute the closed-loop frequency response (cart position output)."""

import numpy as np
import scipy.signal as sig


def compute_closed_loop_response(A_cl, B, w) -> dict:
    """Return dict with w_cl, mag_cl, phase_cl, sys_CL.

    Output is cart position: C_x = [1, 0, ..., 0].
    """
    n = A_cl.shape[0]
    C_x = np.zeros((1, n))
    C_x[0, 0] = 1.0
    D = np.zeros((1, 1))

    sys_CL = sig.lti(A_cl, B.reshape(-1, 1), C_x, D)
    w_cl, mag_cl, phase_cl = sig.bode(sys_CL, w=w)

    return dict(w_cl=w_cl, mag_cl=mag_cl, phase_cl=phase_cl, sys_CL=sys_CL)
