"""Gain scheduling stability verification via eigenvalue analysis."""

import numpy as np
from control.linearization.linearize import linearize
from control.cost_matrices.default_Q import default_Q
from control.cost_matrices.default_R import default_R
from control.riccati.solve_care import solve_riccati


def verify_gain_scheduling_stability(cfg, gs):
    """Verify stability of gain-scheduled controller.

    For each operating point, computes the closed-loop system matrix
    A_cl = A - B @ K and checks eigenvalue stability. Also checks
    stability at interpolated intermediate points.

    Parameters
    ----------
    cfg : SystemConfig
    gs : GainScheduler

    Returns
    -------
    dict with keys:
        'operating_points_deg' : array of deviation angles in degrees
        'all_points_stable' : bool
        'interpolated_all_stable' : bool
        'max_eigenvalue_real' : float (worst-case max Re(eigenvalue))
        'condition_numbers' : P matrix condition numbers at each point
        'eigenvalues_per_point' : list of eigenvalue arrays
    """
    p = cfg.pack()
    q_eq = cfg.equilibrium
    Q = default_Q()
    R = default_R()

    dev_angles = gs.deviation_angles  # radians, sorted
    K_gains = gs.K_gains              # (n_pts, 8)
    n_pts = len(dev_angles)

    operating_points_deg = np.rad2deg(dev_angles)

    eigenvalues_per_point = []
    condition_numbers = []
    all_points_stable = True
    max_eig_real = -np.inf

    # Step 1-3: Check each operating point
    for i in range(n_pts):
        q_op = q_eq.copy()
        q_op[1] += dev_angles[i]

        A, B = linearize(q_op, p)
        K_i = K_gains[i].reshape(1, 8)
        A_cl = A - B @ K_i

        eigs = np.linalg.eigvals(A_cl)
        eigenvalues_per_point.append(eigs)

        max_re = np.max(eigs.real)
        if max_re > max_eig_real:
            max_eig_real = max_re

        if max_re >= 0:
            all_points_stable = False

        # Compute P matrix condition number
        P = solve_riccati(A, B, Q, R)
        cond = np.linalg.cond(P)
        condition_numbers.append(cond)

    # Step 4: Check interpolated stability
    interpolated_all_stable = True
    n_interp = 100

    for i in range(n_pts - 1):
        for j in range(1, n_interp + 1):
            t = j / (n_interp + 1)
            # Interpolated angle
            delta = (1 - t) * dev_angles[i] + t * dev_angles[i + 1]

            # Linearize at interpolated point
            q_interp = q_eq.copy()
            q_interp[1] += delta

            A, B = linearize(q_interp, p)

            # Interpolated gain
            K_interp = (1 - t) * K_gains[i] + t * K_gains[i + 1]
            K_interp_mat = K_interp.reshape(1, 8)

            A_cl = A - B @ K_interp_mat
            eigs = np.linalg.eigvals(A_cl)

            max_re = np.max(eigs.real)
            if max_re > max_eig_real:
                max_eig_real = max_re

            if max_re >= 0:
                interpolated_all_stable = False

    condition_numbers = np.array(condition_numbers)

    return {
        'operating_points_deg': operating_points_deg,
        'all_points_stable': bool(all_points_stable),
        'interpolated_all_stable': bool(interpolated_all_stable),
        'max_eigenvalue_real': float(max_eig_real),
        'condition_numbers': condition_numbers,
        'eigenvalues_per_point': eigenvalues_per_point,
    }
