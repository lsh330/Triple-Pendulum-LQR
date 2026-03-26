"""Compute the step response and time-domain performance metrics."""

import numpy as np
import scipy.signal as sig


def compute_step_response(sys_CL) -> dict:
    """Return dict with t_step, y_step, overshoot, t_settle, y_ss.

    Settling time uses the 2% band around steady-state.
    """
    t_step, y_step = sig.step(sys_CL)

    y_ss = y_step[-1] if len(y_step) > 0 else 0.0

    # Overshoot (%)
    if abs(y_ss) > 1e-12:
        overshoot = (np.max(y_step) - y_ss) / abs(y_ss) * 100.0
    else:
        overshoot = 0.0

    # Settling time (2% band)
    tol = 0.02 * abs(y_ss) if abs(y_ss) > 1e-12 else 0.02
    settled = np.where(np.abs(y_step - y_ss) > tol)[0]
    t_settle = t_step[settled[-1]] if len(settled) > 0 else 0.0

    return dict(t_step=t_step, y_step=y_step, overshoot=overshoot,
                t_settle=t_settle, y_ss=y_ss)
