import numpy as np
from simulation.integrator.rk4_step import rk4_step
from simulation.loop.control_law import compute_control
from simulation.initial_conditions.impulse_response import apply_impulse


def simulate(cfg, K, t_end=10.0, dt=0.002, impulse=0.0, disturbance=None):
    """Run a closed-loop simulation of the cart + triple pendulum.

    Parameters
    ----------
    cfg : SystemConfig
    K : ndarray (1x8)
        LQR gain matrix.
    t_end : float
        Simulation duration in seconds.
    dt : float
        Time step.
    impulse : float
        Initial impulse applied to the cart.
    disturbance : ndarray or None
        External disturbance force array (same length as time array).

    Returns
    -------
    t_arr : ndarray (N,)
    q_arr : ndarray (N, 4)
    dq_arr : ndarray (N, 4)
    u_ctrl_arr : ndarray (N,)
    u_dist_arr : ndarray (N,)
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium
    K_flat = K.flatten()

    N = int(round(t_end / dt)) + 1
    t_arr = np.linspace(0.0, t_end, N)

    q_arr = np.zeros((N, 4))
    dq_arr = np.zeros((N, 4))
    u_ctrl_arr = np.zeros(N)
    u_dist_arr = np.zeros(N)

    # Initial conditions
    q = q_eq.copy()
    dq = np.zeros(4)

    if impulse != 0.0:
        dq = apply_impulse(q_eq, p, impulse)

    q_arr[0] = q
    dq_arr[0] = dq

    for k in range(N - 1):
        # Control
        u_c = compute_control(q, dq, q_eq, K_flat)
        u_ctrl_arr[k] = u_c

        # Disturbance
        u_d = 0.0
        if disturbance is not None:
            u_d = disturbance[k]
        u_dist_arr[k] = u_d

        # Total input (scalar)
        u_total = u_c + u_d

        # Integrate
        q, dq = rk4_step(q, dq, u_total, p, dt)

        q_arr[k + 1] = q
        dq_arr[k + 1] = dq

    # Record control at last step
    u_ctrl_arr[N - 1] = compute_control(q, dq, q_eq, K_flat)
    if disturbance is not None and len(disturbance) >= N:
        u_dist_arr[N - 1] = disturbance[N - 1]

    return t_arr, q_arr, dq_arr, u_ctrl_arr, u_dist_arr
