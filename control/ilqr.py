"""Iterative Linear Quadratic Regulator (iLQR) for the cart + triple pendulum.

iLQR refines a trajectory by iteratively:
1. Forward-simulating with the current control sequence
2. Backward Riccati pass to compute time-varying LQR gains
3. Forward pass with the updated gains
4. Repeating until convergence

This runs offline (not in the real-time JIT loop).
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
    """Linearize and discretize around (q, dq, u) using forward Euler on
    the continuous-time Jacobians.

    Returns discrete A_d (8x8) and B_d (8x1).
    """
    A_q = compute_A_q(q, dq, u, p, eps=None)
    A_dq = compute_A_dq(q, dq, u, p, eps=None)
    B_u = compute_B_u(q, dq, u, p, eps=None)
    A_c, B_c = assemble_state_space(A_q, A_dq, B_u)

    # First-order discretization: A_d = I + dt*A_c, B_d = dt*B_c
    n = A_c.shape[0]
    A_d = np.eye(n) + dt * A_c
    B_d = dt * B_c
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

    # Get the standard LQR gain as warmstart
    K_lqr, _, _, _, _, _ = compute_lqr_gains(cfg)
    K_flat = K_lqr.flatten()

    # Initialize control sequence using LQR policy
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
    for iteration in range(n_iter):
        # --- Backward pass: Riccati recursion ---
        K_seq = np.zeros((N_horizon, n_u, n_x))
        S = Q.copy()  # terminal cost-to-go

        for k in range(N_horizon - 2, -1, -1):
            qk = x_traj[k, :4].copy()
            dqk = x_traj[k, 4:].copy()
            uk = u_traj[k]

            A_d, B_d = _linearize_at(qk, dqk, uk, p, dt)

            # Riccati recursion (discrete-time)
            # Q_xx = Q + A^T S A
            # Q_uu = R + B^T S B
            # Q_ux = B^T S A
            # K_k = Q_uu^{-1} Q_ux
            # S = Q_xx - Q_ux^T K_k  (= Q + A^T S A - K^T Q_uu K)
            Q_xx = Q + A_d.T @ S @ A_d
            Q_uu = R + B_d.T @ S @ B_d
            Q_ux = B_d.T @ S @ A_d

            K_k = np.linalg.solve(Q_uu, Q_ux)  # (1, 8)
            K_seq[k] = K_k

            S = Q_xx - K_k.T @ Q_uu @ K_k

            # Ensure symmetry
            S = 0.5 * (S + S.T)

        # --- Forward pass with new gains ---
        x_new = np.zeros((N_horizon, n_x))
        u_new = np.zeros(N_horizon)
        q = q0.copy()
        dq = dq0.copy()

        for k in range(N_horizon):
            x_new[k, :4] = q
            x_new[k, 4:] = dq

            # Deviation from nominal
            dx = np.empty(n_x)
            dx[:4] = q - x_traj[k, :4]
            dx[4:] = dq - x_traj[k, 4:]

            # Updated control: u = u_nom - K * dx
            u_k = u_traj[k] - (K_seq[k] @ dx)[0]
            u_new[k] = u_k

            if k < N_horizon - 1:
                q, dq = _rk4_step(q, dq, u_k, p, dt)

        x_traj = x_new
        u_traj = u_new

    # Convert gains to output format (N_horizon, 1, 8)
    # The gains K_seq are already in the right shape from the last backward pass
    # But we need to express them as gains around equilibrium for downstream use
    K_traj = K_seq.copy()

    return K_traj, x_traj
