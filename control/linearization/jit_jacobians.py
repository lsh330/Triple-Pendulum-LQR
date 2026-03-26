"""JIT-compiled Jacobian computation for LQR linearization.

Computes all Jacobians (A_q, A_dq, B_u) and assembles the full state-space
matrices A (8x8) and B (8x1) in a single @njit function, eliminating
Python-level loop overhead.
"""
import numpy as np
from numba import njit
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics


@njit(cache=True)
def compute_jacobians_jit(q_eq, dq_eq, u_eq, p):
    """Compute linearized state-space matrices A (8x8) and B (8x1).

    Uses central finite differences with adaptive per-component step sizes.

    Parameters
    ----------
    q_eq : ndarray (4,)
        Equilibrium configuration.
    dq_eq : ndarray (4,)
        Equilibrium velocities (typically zeros).
    u_eq : float
        Equilibrium control input (typically 0.0).
    p : ndarray
        Packed system parameters.

    Returns
    -------
    A : ndarray (8, 8)
    B : ndarray (8, 1)
    """
    n = 4

    # --- A_q: central diff of forward_dynamics w.r.t. q (4x4) ---
    A_q = np.empty((n, n))
    for j in range(n):
        h = 1.49e-8 * max(1.0, abs(q_eq[j]))
        q_plus = q_eq.copy()
        q_minus = q_eq.copy()
        q_plus[j] += h
        q_minus[j] -= h
        ddq_plus = forward_dynamics(q_plus, dq_eq, u_eq, p)
        ddq_minus = forward_dynamics(q_minus, dq_eq, u_eq, p)
        for i in range(n):
            A_q[i, j] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h)

    # --- A_dq: central diff of forward_dynamics w.r.t. dq (4x4) ---
    A_dq = np.empty((n, n))
    for j in range(n):
        h = 1.49e-8 * max(1.0, abs(dq_eq[j]))
        dq_plus = dq_eq.copy()
        dq_minus = dq_eq.copy()
        dq_plus[j] += h
        dq_minus[j] -= h
        ddq_plus = forward_dynamics(q_eq, dq_plus, u_eq, p)
        ddq_minus = forward_dynamics(q_eq, dq_minus, u_eq, p)
        for i in range(n):
            A_dq[i, j] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h)

    # --- B_u: central diff of forward_dynamics w.r.t. u (4x1) ---
    h_u = 1.49e-8 * max(1.0, abs(u_eq))
    ddq_plus = forward_dynamics(q_eq, dq_eq, u_eq + h_u, p)
    ddq_minus = forward_dynamics(q_eq, dq_eq, u_eq - h_u, p)
    B_u = np.empty((n, 1))
    for i in range(n):
        B_u[i, 0] = (ddq_plus[i] - ddq_minus[i]) / (2.0 * h_u)

    # --- Assemble A (8x8) and B (8x1) ---
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, 1))

    # Top-right block: identity
    for i in range(n):
        A[i, n + i] = 1.0

    # Bottom-left block: A_q
    for i in range(n):
        for j in range(n):
            A[n + i, j] = A_q[i, j]

    # Bottom-right block: A_dq
    for i in range(n):
        for j in range(n):
            A[n + i, n + j] = A_dq[i, j]

    # Bottom of B: B_u
    for i in range(n):
        B[n + i, 0] = B_u[i, 0]

    return A, B
