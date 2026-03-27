"""Linearize the triple-pendulum system about an operating point.

Uses analytical Jacobians (exact dG/dq, dM/dq, M^{-1}) for A_q and B_u,
with numerical central differences only for A_dq (Coriolis velocity term).
Falls back to fully numerical JIT Jacobians if analytical computation fails.
"""

import numpy as np
from control.linearization.analytical_jacobian import compute_analytical_jacobians
from control.linearization.jit_jacobians import compute_jacobians_jit


def linearize(q_eq, p, eps=None, method="analytical"):
    """Linearize the system about the equilibrium point.

    Returns (A, B) where A is 8x8 and B is 8x1.
    Logs analytical vs numerical Jacobian comparison at DEBUG level.
    """
    from utils.logger import get_logger
    log = get_logger()

    if method == "analytical":
        try:
            A, B = compute_analytical_jacobians(q_eq, p)

            # Self-verification: compare with numerical at DEBUG level
            if log.isEnabledFor(10):  # DEBUG = 10
                dq_eq = np.zeros(4)
                u_eq = 0.0
                A_num, B_num = compute_jacobians_jit(q_eq, dq_eq, u_eq, p)
                max_diff_A = np.max(np.abs(A - A_num))
                max_diff_B = np.max(np.abs(B - B_num))
                log.debug("Linearization self-check: max|A_an-A_fd|=%.2e, max|B_an-B_fd|=%.2e",
                         max_diff_A, max_diff_B)

            return A, B
        except (np.linalg.LinAlgError, ValueError) as e:
            log.warning("Analytical Jacobian failed (%s), falling back to numerical", e)

    dq_eq = np.zeros(4)
    u_eq = 0.0
    A, B = compute_jacobians_jit(q_eq, dq_eq, u_eq, p)
    return A, B
