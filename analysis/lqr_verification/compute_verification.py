"""Compute all LQR verification metrics."""

import numpy as np
import scipy.signal as sig


def compute_lqr_verification(t, q, dq, q_eq, K, A, B, P, Q, R, freq_w=None):
    """
    Compute metrics that prove LQR is working correctly.

    Returns dict with:
      - P_eigenvalues: eigenvalues of Riccati solution (must all be > 0)
      - lyapunov_V: V(t) = z(t)' P z(t)  (must be monotonically decreasing)
      - cost_state: x'Qx at each timestep
      - cost_control: u'Ru at each timestep
      - cost_cumulative: integral of (x'Qx + u'Ru) over time
      - return_difference: |1 + L(jw)| (must be >= 1 for LQR, Kalman inequality)
      - return_diff_w: frequency array for return difference
      - nyquist_encirclements: number of CCW encirclements of (-1,0)
      - n_unstable_ol: number of unstable open-loop poles
      - nyquist_criterion_ok: True if encirclements == n_unstable_ol
      - A_cl_eigenvalues: closed-loop eigenvalues
      - A_cl_damping: damping ratios for each CL pole
      - A_cl_wn: natural frequencies for each CL pole
    """
    if freq_w is None:
        freq_w = np.logspace(-2, 3, 3000)

    N = len(t)
    dt = t[1] - t[0]
    K_flat = K.flatten()

    # --- Riccati P matrix ---
    P_eig = np.linalg.eigvalsh(P)

    # --- Lyapunov function and cost ---
    z = np.zeros((N, 8))
    z[:, :4] = q - q_eq
    z[:, 4:] = dq

    lyapunov_V = np.array([z_i @ P @ z_i for z_i in z])

    cost_state = np.array([z_i @ Q @ z_i for z_i in z])
    u_ctrl = np.array([-K_flat @ z_i for z_i in z])
    cost_control = np.array([u_i * R[0, 0] * u_i for u_i in u_ctrl])
    cost_total = cost_state + cost_control
    cost_cumulative = np.cumsum(cost_total) * dt

    # --- Return difference: |1 + L(jw)| >= 1 (Kalman inequality) ---
    C_L = K.reshape(1, -1)
    D_L = np.zeros((1, 1))
    sys_L = sig.lti(A, B.reshape(-1, 1), C_L, D_L)
    w_rd, H_rd = sig.freqresp(sys_L, w=freq_w)
    L_jw = H_rd.flatten()
    return_diff = np.abs(1.0 + L_jw)

    # --- Nyquist encirclement count ---
    # Count net CW encirclements of (-1,0) for w: 0 -> inf
    # Using winding number: integral of d(angle) / (2*pi)
    shifted = L_jw + 1.0  # shift so we count encirclements of origin
    angles = np.angle(shifted)
    d_angle = np.diff(np.unwrap(angles))
    net_angle = np.sum(d_angle)
    # For w>0 only, multiply by 2 for full contour (symmetry for real systems)
    # CW encirclements (negative winding) of (-1,0) for full Nyquist = N
    # Nyquist criterion: N_cw = P_ol (# unstable OL poles) for stability
    full_winding = net_angle / np.pi  # half contour gives half the winding
    n_encirclements_cw = -int(np.round(full_winding))  # CW = negative math convention

    n_unstable_ol = np.sum(np.linalg.eigvals(A).real > 0)
    nyquist_ok = (n_encirclements_cw == n_unstable_ol)

    # --- Closed-loop pole analysis ---
    A_cl = A - B @ K
    cl_eig = np.linalg.eigvals(A_cl)
    wn = np.abs(cl_eig)
    damping = np.where(wn > 1e-10, -cl_eig.real / wn, 1.0)

    return {
        "P_eigenvalues": P_eig,
        "lyapunov_V": lyapunov_V,
        "cost_state": cost_state,
        "cost_control": cost_control,
        "cost_cumulative": cost_cumulative,
        "return_diff_w": w_rd,
        "return_difference": return_diff,
        "L_jw_full": L_jw,
        "n_encirclements_cw": n_encirclements_cw,
        "n_unstable_ol": int(n_unstable_ol),
        "nyquist_criterion_ok": nyquist_ok,
        "A_cl_eigenvalues": cl_eig,
        "A_cl_damping": damping,
        "A_cl_wn": wn,
        "t": t,
    }
