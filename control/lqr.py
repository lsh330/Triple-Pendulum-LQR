"""End-to-end LQR gain computation with input validation."""

import numpy as np
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q, equilibrium_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati
from control.gain_computation.compute_K import compute_K


def _validate_cost_matrices(Q, R, n_x, n_u):
    """Validate Q is positive semi-definite and R is positive definite."""
    if Q.shape != (n_x, n_x):
        raise ValueError(f"Q must be ({n_x},{n_x}), got {Q.shape}")
    if R.shape != (n_u, n_u):
        raise ValueError(f"R must be ({n_u},{n_u}), got {R.shape}")

    # Q must be symmetric positive semi-definite
    if not np.allclose(Q, Q.T, atol=1e-12):
        raise ValueError("Q must be symmetric")
    eig_Q = np.linalg.eigvalsh(Q)
    if np.min(eig_Q) < -1e-10:
        raise ValueError(f"Q must be positive semi-definite (min eigenvalue={np.min(eig_Q):.2e})")

    # R must be symmetric positive definite
    if not np.allclose(R, R.T, atol=1e-12):
        raise ValueError("R must be symmetric")
    eig_R = np.linalg.eigvalsh(R)
    if np.min(eig_R) <= 0:
        raise ValueError(f"R must be positive definite (min eigenvalue={np.min(eig_R):.2e})")


def compute_lqr_gains(cfg, Q=None, R=None):
    """Compute LQR gains for the triple inverted pendulum.

    Validates cost matrices and controllability before solving CARE.
    Returns K, A, B, P, Q, R.
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium

    A, B = linearize(q_eq, p)

    if Q is None:
        Q = default_Q()
    if R is None:
        R = default_R()

    n_x = A.shape[0]
    n_u = B.shape[1] if B.ndim == 2 else 1
    _validate_cost_matrices(Q, R, n_x, n_u)

    P = solve_riccati(A, B, Q, R)
    K = compute_K(R, B, P)

    return K, A, B, P, Q, R


def compute_all_lqr_gains(cfg, Q_override=None, R_override=None):
    """Compute LQR gains for all 8 equilibrium configurations.

    For each configuration the system is linearized at its equilibrium and
    CARE is solved with a configuration-adapted cost matrix (upright links
    penalized more heavily). A ValueError or LinAlgError for any configuration
    is caught and stored as None.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration. The internal _target_eq field is temporarily
        modified per configuration and restored before returning.
    Q_override : np.ndarray or None
        If provided, use this Q for all configurations instead of
        equilibrium_Q(). Must be (8,8) positive semi-definite.
    R_override : np.ndarray or None
        If provided, use this R for all configurations. Must be (1,1)
        positive definite.

    Returns
    -------
    dict
        Maps config_name -> result_dict or None.
        result_dict keys:
            'K'              : gain matrix (1,8)
            'A'              : state matrix (8,8)
            'B'              : input matrix (8,1)
            'P'              : Riccati solution (8,8)
            'Q'              : cost matrix (8,8)
            'R'              : input cost (1,1)
            'q_eq'           : equilibrium configuration (4,)
            'V_eq'           : physical potential energy at equilibrium (float, J)
            'poles'          : closed-loop eigenvalues (8,)
            'is_controllable': True (False configs are stored as None)
            'is_stable'      : bool, all real parts < 0
    """
    from parameters.equilibrium import all_equilibria, equilibrium_potential_energy
    from control.closed_loop import compute_closed_loop

    R_mat = R_override if R_override is not None else default_R()

    results = {}
    original_eq = cfg._target_eq

    for name, q_eq in all_equilibria().items():
        cfg._target_eq = name
        try:
            p = cfg.pack()
            A, B = linearize(q_eq, p)

            Q = Q_override if Q_override is not None else equilibrium_Q(name)

            n_x = A.shape[0]
            n_u = B.shape[1] if B.ndim == 2 else 1
            _validate_cost_matrices(Q, R_mat, n_x, n_u)

            P = solve_riccati(A, B, Q, R_mat)
            K = compute_K(R_mat, B, P)
            cl = compute_closed_loop(A, B, K)

            V_eq = equilibrium_potential_energy(cfg, name)

            results[name] = {
                'K': K,
                'A': A,
                'B': B,
                'P': P,
                'Q': Q,
                'R': R_mat,
                'q_eq': q_eq,
                'V_eq': V_eq,
                'poles': cl['poles'],
                'is_controllable': True,
                'is_stable': cl['is_stable'],
            }
        except (ValueError, np.linalg.LinAlgError):
            results[name] = None
        finally:
            cfg._target_eq = original_eq

    return results
