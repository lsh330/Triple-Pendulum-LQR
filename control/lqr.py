from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati
from control.gain_computation.compute_K import compute_K


def compute_lqr_gains(cfg, Q=None, R=None):
    """Compute LQR gains for the triple inverted pendulum.

    Returns K, A, B, P, Q, R
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium

    A, B = linearize(q_eq, p)

    if Q is None:
        Q = default_Q()
    if R is None:
        R = default_R()

    P = solve_riccati(A, B, Q, R)
    K = compute_K(R, B, P)

    return K, A, B, P, Q, R
