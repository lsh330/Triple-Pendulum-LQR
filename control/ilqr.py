"""Iterative Linear Quadratic Regulator (iLQR) for the cart + triple pendulum.

iLQR refines a trajectory by iteratively:
1. Forward-simulating with the current control sequence
2. Backward Riccati pass to compute time-varying LQR gains
3. Forward pass with the updated gains
4. Repeating until convergence (cost reduction < tolerance)

Uses matrix exponential for accurate discretization (C4 fix).
"""

import numpy as np
from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
from control.linearization.jacobian_q import compute_A_q
from control.linearization.jacobian_dq import compute_A_dq
from control.linearization.jacobian_u import compute_B_u
from control.linearization.state_space import assemble_state_space
from control.lqr import compute_lqr_gains
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R


def _rk4_step(q, dq, u, p, dt):
    """Single RK4 step (pure Python/numpy, no JIT needed)."""
    state = np.empty(8)
    state[:4] = q
    state[4:] = dq

    def deriv(s, u_val):
        ds = np.empty(8)
        ds[:4] = s[4:]
        ds[4:] = forward_dynamics(s[:4].copy(), s[4:].copy(), u_val, p)
        return ds

    k1 = deriv(state, u)
    k2 = deriv(state + 0.5 * dt * k1, u)
    k3 = deriv(state + 0.5 * dt * k2, u)
    k4 = deriv(state + dt * k3, u)

    state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state_new[:4].copy(), state_new[4:].copy()


def _linearize_at(q, dq, u, p, dt):
    """Linearize and discretize around (q, dq, u) using matrix exponential.

    Uses the exact discretization:
        A_d = expm(A_c * dt)
        B_d = A_c^{-1} (A_d - I) B_c  (or dt*B_c for small dt fallback)

    Returns discrete A_d (8x8) and B_d (8x1).
    """
    from scipy.linalg import expm

    A_q = compute_A_q(q, dq, u, p, eps=None)
    A_dq = compute_A_dq(q, dq, u, p, eps=None)
    B_u = compute_B_u(q, dq, u, p, eps=None)
    A_c, B_c = assemble_state_space(A_q, A_dq, B_u)

    n = A_c.shape[0]
    A_d = expm(A_c * dt)

    # Exact B discretization via augmented matrix exponential
    # [A_c B_c; 0 0] * dt -> expm gives [A_d, B_d_integrated; 0, I]
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = A_c * dt
    aug[:n, n:] = B_c * dt
    aug_exp = expm(aug)
    B_d = aug_exp[:n, n:].copy()

    return A_d, B_d


def compute_ilqr_gains(cfg, q0, dq0, N_horizon=500, dt=0.001, n_iter=10):
    """Compute time-varying iLQR gains along a trajectory.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration.
    q0 : ndarray (4,)
        Initial joint configuration.
    dq0 : ndarray (4,)
        Initial joint velocities.
    N_horizon : int
        Number of time steps in the planning horizon.
    dt : float
        Integration time step.
    n_iter : int
        Number of iLQR iterations.

    Returns
    -------
    K_traj : ndarray (N_horizon, 1, 8)
        Time-varying gain matrices along the trajectory.
    x_traj : ndarray (N_horizon, 8)
        Nominal state trajectory.
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium
    n_x = 8
    n_u = 1

    Q = default_Q()
    R = default_R()

    K_lqr, _, _, P_lqr, _, _ = compute_lqr_gains(cfg)
    K_flat = K_lqr.flatten()

    x_traj = np.zeros((N_horizon, n_x))
    u_traj = np.zeros(N_horizon)

    # Forward simulate with LQR to get initial trajectory
    q = q0.copy()
    dq = dq0.copy()
    for k in range(N_horizon):
        x_traj[k, :4] = q
        x_traj[k, 4:] = dq
        z = np.empty(n_x)
        z[:4] = q - q_eq
        z[4:] = dq
        u_k = -K_flat @ z
        u_traj[k] = u_k
        if k < N_horizon - 1:
            q, dq = _rk4_step(q, dq, u_k, p, dt)

    # iLQR iterations
    prev_cost = np.inf
    converged = False

    for iteration in range(n_iter):
        # --- Backward pass: Riccati recursion ---
        K_seq = np.zeros((N_horizon, n_u, n_x))
        S = P_lqr.copy()  # Terminal cost = CARE solution (Lyapunov stability guarantee)

        for k in range(N_horizon - 2, -1, -1):
            qk = x_traj[k, :4].copy()
            dqk = x_traj[k, 4:].copy()
            uk = u_traj[k]

            A_d, B_d = _linearize_at(qk, dqk, uk, p, dt)

            Q_xx = Q + A_d.T @ S @ A_d
            Q_uu = R + B_d.T @ S @ B_d
            Q_ux = B_d.T @ S @ A_d

            K_k = np.linalg.solve(Q_uu, Q_ux)
            K_seq[k] = K_k

            S = Q_xx - K_k.T @ Q_uu @ K_k
            S = 0.5 * (S + S.T)

        # --- Forward pass with new gains ---
        x_new = np.zeros((N_horizon, n_x))
        u_new = np.zeros(N_horizon)
        q = q0.copy()
        dq = dq0.copy()
        total_cost = 0.0

        for k in range(N_horizon):
            x_new[k, :4] = q
            x_new[k, 4:] = dq

            dx = np.empty(n_x)
            dx[:4] = q - x_traj[k, :4]
            dx[4:] = dq - x_traj[k, 4:]

            u_k = u_traj[k] - (K_seq[k] @ dx)[0]
            u_new[k] = u_k

            # Accumulate cost
            z_k = np.empty(n_x)
            z_k[:4] = q - q_eq
            z_k[4:] = dq
            total_cost += float(z_k @ Q @ z_k + u_k * R[0, 0] * u_k) * dt

            if k < N_horizon - 1:
                q, dq = _rk4_step(q, dq, u_k, p, dt)

        # Convergence check
        cost_reduction = (prev_cost - total_cost) / max(abs(prev_cost), 1e-10)
        if iteration > 0 and abs(cost_reduction) < 1e-4:
            converged = True

        x_traj = x_new
        u_traj = u_new
        prev_cost = total_cost

        if converged:
            break

    K_traj = K_seq.copy()
    return K_traj, x_traj
