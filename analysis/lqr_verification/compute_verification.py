"""Compute all LQR verification metrics."""

import numpy as np
import scipy.signal as sig

from parameters.config import SystemConfig
from control.lqr import compute_lqr_gains


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


def compute_monte_carlo_robustness(cfg, n_samples=20, perturbation=0.10, seed=42):
    """Compute Monte Carlo Bode and pole data with +/-10% mass perturbation.

    Returns dict with:
      - mc_bode_w: frequency array
      - mc_bode_mag: list of magnitude arrays (dB) for each perturbed system
      - mc_cl_poles: list of closed-loop pole arrays for each perturbed system
      - nominal_bode_mag: nominal magnitude array (dB)
      - nominal_cl_poles: nominal closed-loop poles
    """
    rng = np.random.default_rng(seed)
    freq_w = np.logspace(-2, 3, 500)

    # Nominal system
    K_nom, A_nom, B_nom, _, _, _ = compute_lqr_gains(cfg)
    C_nom = K_nom.reshape(1, -1)
    D_nom = np.zeros((1, 1))
    sys_nom = sig.lti(A_nom, B_nom.reshape(-1, 1), C_nom, D_nom)
    w_nom, H_nom = sig.freqresp(sys_nom, w=freq_w)
    mag_nom_dB = 20 * np.log10(np.abs(H_nom.flatten()) + 1e-30)
    A_cl_nom = A_nom - B_nom @ K_nom
    poles_nom = np.linalg.eigvals(A_cl_nom)

    mc_bode_mag = []
    mc_cl_poles = []

    for _ in range(n_samples):
        # Perturb masses by +/-perturbation
        scale_mc = 1.0 + rng.uniform(-perturbation, perturbation)
        scale_m1 = 1.0 + rng.uniform(-perturbation, perturbation)
        scale_m2 = 1.0 + rng.uniform(-perturbation, perturbation)
        scale_m3 = 1.0 + rng.uniform(-perturbation, perturbation)

        cfg_pert = SystemConfig(
            mc=cfg.mc * scale_mc,
            m1=cfg.m1 * scale_m1,
            m2=cfg.m2 * scale_m2,
            m3=cfg.m3 * scale_m3,
            L1=cfg.L1, L2=cfg.L2, L3=cfg.L3,
            g=cfg._phys.g,
        )

        try:
            K_p, A_p, B_p, _, _, _ = compute_lqr_gains(cfg_pert)
            C_p = K_p.reshape(1, -1)
            D_p = np.zeros((1, 1))
            sys_p = sig.lti(A_p, B_p.reshape(-1, 1), C_p, D_p)
            _, H_p = sig.freqresp(sys_p, w=freq_w)
            mag_p_dB = 20 * np.log10(np.abs(H_p.flatten()) + 1e-30)
            A_cl_p = A_p - B_p @ K_p
            poles_p = np.linalg.eigvals(A_cl_p)
            mc_bode_mag.append(mag_p_dB)
            mc_cl_poles.append(poles_p)
        except Exception:
            # Skip failed perturbations
            continue

    return {
        "mc_bode_w": w_nom,
        "mc_bode_mag": mc_bode_mag,
        "mc_cl_poles": mc_cl_poles,
        "nominal_bode_mag": mag_nom_dB,
        "nominal_cl_poles": poles_nom,
    }
