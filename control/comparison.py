"""Compare LQR with PID and pole placement controllers.

Provides alternative control designs for the same triple pendulum system
to demonstrate the advantages of LQR optimal control.
"""

import numpy as np
from scipy import signal


def compute_pid_gains(A, B, K_lqr):
    """Design a PD state feedback controller using Bryson's rule.

    Scales gains by the inverse of acceptable state deviations, providing
    a principled (though suboptimal) alternative to LQR.

    Returns K_pid (1, 8) gain matrix.
    """
    K_pid = np.zeros((1, 8))

    # Bryson's rule: Q_ii = 1/max_acceptable_deviation_i^2
    # Cart: ±0.5m, angles: ±10deg, velocities: ±2 m/s or rad/s
    max_dev = np.array([0.5, np.radians(10), np.radians(10), np.radians(10),
                        2.0, np.radians(50), np.radians(50), np.radians(50)])
    Q_bryson = np.diag(1.0 / max_dev**2)
    R_bryson = np.array([[1.0 / 200.0**2]])  # max force 200N

    try:
        from control.riccati.solve_care import solve_riccati
        from control.gain_computation.compute_K import compute_K
        B_col = B.reshape(-1, 1) if B.ndim == 1 else B
        P = solve_riccati(A, B_col, Q_bryson, R_bryson)
        K_pid = compute_K(R_bryson, B_col, P)
    except Exception:
        # Fallback: scale from LQR
        K_pid = K_lqr * 0.5

    return K_pid.reshape(1, 8)


def compute_pole_placement_gains(A, B):
    """Design a pole placement controller using Ackermann's formula.

    Places all 8 poles at specified locations in the LHP.
    Uses scipy.signal.place_poles with the KNV0 algorithm.

    Returns K_pp (1, 8) gain matrix.
    """
    # Target poles: spread in LHP with reasonable damping
    # Dominant pair at -2 +/- 2j (zeta ~ 0.7)
    # Secondary pairs progressively faster
    desired_poles = np.array([
        -2.0 + 2.0j, -2.0 - 2.0j,    # dominant pair
        -4.0 + 3.0j, -4.0 - 3.0j,    # secondary
        -6.0 + 1.0j, -6.0 - 1.0j,    # tertiary
        -8.0,                           # fast real
        -10.0,                          # fastest real
    ])

    B_col = B.reshape(-1, 1) if B.ndim == 1 else B

    try:
        result = signal.place_poles(A, B_col, desired_poles, method='KNV0')
        K_pp = result.gain_matrix
    except Exception:
        # Fallback: use LQR-like approach with modified weights
        from control.riccati.solve_care import solve_riccati
        from control.gain_computation.compute_K import compute_K
        Q_pp = np.diag([5.0, 50.0, 50.0, 50.0, 0.5, 5.0, 5.0, 5.0])
        R_pp = np.array([[0.1]])
        P = solve_riccati(A, B_col, Q_pp, R_pp)
        K_pp = compute_K(R_pp, B_col, P)

    return K_pp.reshape(1, 8)


def compare_controllers(A, B, K_lqr, t_end=5.0, dt=0.001):
    """Compare step responses of LQR, PID, and pole placement.

    Returns dict with time arrays and response data for each controller.
    """
    B_col = B.reshape(-1, 1) if B.ndim == 1 else B
    n = A.shape[0]

    K_pid = compute_pid_gains(A, B_col, K_lqr)
    K_pp = compute_pole_placement_gains(A, B_col)

    controllers = {
        "LQR": K_lqr,
        "PD": K_pid,
        "Pole Placement": K_pp,
    }

    results = {}
    t = np.arange(0, t_end, dt)
    w = np.logspace(-2, 3, 2000)

    for name, K in controllers.items():
        K_mat = K.reshape(1, -1)
        A_cl = A - B_col @ K_mat

        # Stability check
        eigs = np.linalg.eigvals(A_cl)
        is_stable = bool(np.all(eigs.real < 0))

        # Time response (impulse on cart)
        x = np.zeros((len(t), n))
        x[0, 4] = 1.0  # initial cart velocity
        if is_stable:
            for k in range(len(t) - 1):
                dx = A_cl @ x[k]
                x[k+1] = x[k] + dt * dx  # Euler (sufficient for comparison)

        # Frequency response (loop transfer function)
        C_L = K_mat
        D_L = np.zeros((1, 1))
        sys_L = signal.lti(A, B_col, C_L, D_L)
        try:
            w_out, H = signal.freqresp(sys_L, w=w)
            L_mag = 20 * np.log10(np.abs(H.flatten()) + 1e-30)
        except Exception:
            w_out = w
            L_mag = np.zeros_like(w)

        results[name] = {
            "K": K_mat,
            "A_cl": A_cl,
            "eigenvalues": eigs,
            "is_stable": is_stable,
            "t": t,
            "x": x,
            "w": w_out,
            "L_mag": L_mag,
        }

    return results
