"""Lyapunov-based ROA boundary estimation for form-switching.

Method
------
1. Sample N random initial conditions inside a ball of radius *max_angle_deg*
   around the target equilibrium.
2. Simulate each IC forward for *t_horizon* seconds under LQR control.
3. Record the Lyapunov value V0 = z0^T P z0 for ICs that converge.
4. Estimate rho = safety_factor * max(V0 over converged ICs).
5. Return switching thresholds:
     rho_in  = 0.5 * rho   (enter LQR catch when V < rho_in)
     rho_out = 0.8 * rho   (exit back to swing-up when V > rho_out)

W1 수정사항
-----------
- 고정 ``n_samples=300`` 에서 Wilson-score 95% CI 폭 < 0.05 적응형 확장으로 변경.
  (최소 300, 최대 2000 샘플)
- ``u_max`` 를 ``cfg.actuator_saturation`` (기본 200 N) 에서 읽도록 변경.
- ``_roa_simulate_one`` 내부 하드코딩 200 N → 동적 인자 ``u_sat`` 로 교체.
- Wilson CI 계산은 ``analysis.roa_utils`` 모듈로 통일.
"""

import numpy as np
from numba import njit, prange

from core.angle_utils import angle_wrap
from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast
from analysis.roa_utils import get_u_max, wilson_ci_width


@njit(cache=True, fastmath=True, boundscheck=False)
def _roa_simulate_one(
    sq0_init, sq1_init, sq2_init, sq3_init,
    sdq0_init, sdq1_init, sdq2_init, sdq3_init,
    q_eq, K_flat, p, N_steps, dt, convergence_threshold,
    u_sat=200.0,
):
    """Simulate one ROA sample under LQR for N_steps. Zero Python overhead.

    This @njit kernel eliminates the per-RK4-step Python interpreter overhead
    that was incurred when the inner loop ran as a Python for loop calling a
    Numba function repeatedly.  At n_samples=300 and N_steps=3000 the outer
    Python loop dispatches 300 JIT calls instead of 900 000.

    Parameters
    ----------
    sq0_init..sdq3_init : float
        Initial scalar state.
    q_eq : (4,) float64
        Equilibrium configuration.
    K_flat : (8,) float64
        Flattened LQR gain vector.
    p : (13,) float64
        Packed system parameters.
    N_steps : int
        Number of integration steps.
    dt : float
        Integration timestep (s).
    convergence_threshold : float
        Final |z| < threshold -> converged.
    u_sat : float
        W1: 액추에이터 포화 한계 (N). cfg.actuator_saturation 에서 전달.

    Returns
    -------
    (converged : bool, sq0_f, sq1_f, sq2_f, sq3_f,
                       sdq0_f, sdq1_f, sdq2_f, sdq3_f)
    """
    sq0 = sq0_init; sq1 = sq1_init; sq2 = sq2_init; sq3 = sq3_init
    sdq0 = sdq0_init; sdq1 = sdq1_init; sdq2 = sdq2_init; sdq3 = sdq3_init

    for _ in range(N_steps):
        # LQR control: u = -K * z  (with angle wrapping)
        ez0 = sq0 - q_eq[0]
        ez1 = angle_wrap(sq1 - q_eq[1])
        ez2 = angle_wrap(sq2 - q_eq[2])
        ez3 = angle_wrap(sq3 - q_eq[3])
        u = -(K_flat[0] * ez0 + K_flat[1] * ez1 + K_flat[2] * ez2
              + K_flat[3] * ez3 + K_flat[4] * sdq0 + K_flat[5] * sdq1
              + K_flat[6] * sdq2 + K_flat[7] * sdq3)

        # W1: cfg.actuator_saturation 기반 포화 적용 (하드코딩 200 N 제거)
        if u > u_sat:
            u = u_sat
        elif u < -u_sat:
            u = -u_sat

        sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u, p, dt
        )

        # NaN divergence check
        if sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3:
            return (False,
                    sq0, sq1, sq2, sq3,
                    sdq0, sdq1, sdq2, sdq3)

    # Final convergence check: L2 norm of error state
    ez0 = sq0 - q_eq[0]
    ez1 = angle_wrap(sq1 - q_eq[1])
    ez2 = angle_wrap(sq2 - q_eq[2])
    ez3 = angle_wrap(sq3 - q_eq[3])
    norm_z = (ez0 * ez0 + ez1 * ez1 + ez2 * ez2 + ez3 * ez3
              + sdq0 * sdq0 + sdq1 * sdq1 + sdq2 * sdq2 + sdq3 * sdq3) ** 0.5
    converged = norm_z < convergence_threshold
    return (converged,
            sq0, sq1, sq2, sq3,
            sdq0, sdq1, sdq2, sdq3)


@njit(cache=True, parallel=True, fastmath=True, boundscheck=False)
def _roa_batch_lyapunov(
    n_samples,
    sq0_arr, sq1_arr, sq2_arr, sq3_arr,
    sdq0_arr, sdq1_arr, sdq2_arr, sdq3_arr,
    q_eq, K_flat, P_mat, p,
    N_steps, dt, convergence_threshold, u_sat,
):
    """Parallel batch ROA + Lyapunov value computation.

    Combines _roa_simulate_one and V0=z^T P z in a single parallel kernel.
    Each sample is independent -> prange safe.

    Parameters
    ----------
    n_samples : int
    sq0_arr .. sdq3_arr : (n_samples,) float64
        Initial states for each sample.
    q_eq : (4,) float64
    K_flat : (8,) float64
    P_mat : (8, 8) float64
        Lyapunov matrix for V0 = z^T P z.
    p : (13,) float64
    N_steps : int
    dt : float
    convergence_threshold : float
    u_sat : float

    Returns
    -------
    converged : (n_samples,) bool
    V0_arr : (n_samples,) float64  — Lyapunov value at IC
    """
    converged = np.zeros(n_samples, dtype=np.bool_)
    V0_arr = np.zeros(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        sq0 = sq0_arr[i]; sq1 = sq1_arr[i]
        sq2 = sq2_arr[i]; sq3 = sq3_arr[i]
        sdq0 = sdq0_arr[i]; sdq1 = sdq1_arr[i]
        sdq2 = sdq2_arr[i]; sdq3 = sdq3_arr[i]

        # V0 = z^T P z  (inline, no allocation)
        z0 = sq0 - q_eq[0]
        z1 = angle_wrap(sq1 - q_eq[1])
        z2 = angle_wrap(sq2 - q_eq[2])
        z3 = angle_wrap(sq3 - q_eq[3])
        z4 = sdq0; z5 = sdq1; z6 = sdq2; z7 = sdq3
        z = (z0, z1, z2, z3, z4, z5, z6, z7)
        V0 = 0.0
        for r in range(8):
            for c in range(8):
                V0 += P_mat[r, c] * z[r] * z[c]
        V0_arr[i] = V0

        # Simulate N_steps under LQR
        for _ in range(N_steps):
            ez0 = sq0 - q_eq[0]
            ez1 = angle_wrap(sq1 - q_eq[1])
            ez2 = angle_wrap(sq2 - q_eq[2])
            ez3 = angle_wrap(sq3 - q_eq[3])
            u = -(K_flat[0]*ez0 + K_flat[1]*ez1 + K_flat[2]*ez2
                  + K_flat[3]*ez3 + K_flat[4]*sdq0 + K_flat[5]*sdq1
                  + K_flat[6]*sdq2 + K_flat[7]*sdq3)
            if u > u_sat:
                u = u_sat
            elif u < -u_sat:
                u = -u_sat

            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
                sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u, p, dt
            )

            if sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3:
                break
        else:
            # Only check convergence if loop completed without NaN
            ez0 = sq0 - q_eq[0]
            ez1 = angle_wrap(sq1 - q_eq[1])
            ez2 = angle_wrap(sq2 - q_eq[2])
            ez3 = angle_wrap(sq3 - q_eq[3])
            norm_z = (ez0*ez0 + ez1*ez1 + ez2*ez2 + ez3*ez3
                      + sdq0*sdq0 + sdq1*sdq1 + sdq2*sdq2 + sdq3*sdq3) ** 0.5
            if norm_z < convergence_threshold:
                converged[i] = True

    return converged, V0_arr


def _halton_precompute(n_max, dim=7, seed=0):
    """Pre-generate Halton sequence for up to n_max samples.

    Dimensions: [cart, theta1, theta2, theta3, dq0, dq1, dq2, dq3]
    (dim=7 for 3 angles + 4 velocities; cart uses dim 0 separately)
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19][:dim]
    result = np.empty((n_max, dim))
    for d in range(dim):
        base = primes[d]
        for i in range(n_max):
            n = i + 1 + seed
            f = 1.0
            r = 0.0
            while n > 0:
                f /= base
                r += f * (n % base)
                n //= base
            result[i, d] = r
    return result


def estimate_lyapunov_roa(
    cfg,
    K: np.ndarray,
    P: np.ndarray,
    q_eq: np.ndarray,
    n_samples: int = 500,
    max_angle_deg: float = 30.0,
    t_horizon: float = 3.0,
    dt: float = 0.001,
    convergence_threshold: float = 0.1,
    safety_factor: float = 0.8,
    target_ci_width: float = 0.05,
    n_min: int = 300,
    n_max: int = 2000,
) -> dict:
    """Estimate the ROA as the Lyapunov level set {z : z^T P z < rho}.

    Uses Monte Carlo forward simulation with a fixed-gain LQR to classify
    initial conditions as converging or diverging, then fits the largest
    certified level set.

    The inner simulation loop is delegated to the @njit kernel
    ``_roa_simulate_one``, which eliminates Python overhead for all N_steps
    RK4 calls inside each sample (300 JIT dispatches instead of 900 000 for
    the default n_samples=300, t_horizon=3 s, dt=0.001 s configuration).

    Parameters
    ----------
    cfg : SystemConfig
        System configuration.
    K : np.ndarray, shape (1, 8) or (8,)
        LQR gain matrix.
    P : np.ndarray, shape (8, 8)
        Lyapunov/Riccati matrix from CARE solution.
    q_eq : np.ndarray, shape (4,)
        Equilibrium configuration [x, theta1, theta2, theta3].
    n_samples : int
        초기 배치 샘플 수.
    max_angle_deg : float
        Maximum initial angle deviation from equilibrium (degrees).
    t_horizon : float
        Forward simulation horizon per sample (seconds).
    dt : float
        Integration timestep (seconds).
    convergence_threshold : float
        |z_final| < threshold -> considered converged.
    safety_factor : float
        rho = safety_factor * rho_max_empirical (< 1 for robustness margin).
    target_ci_width : float
        W1: Wilson-score 95% CI 목표 폭 (기본 0.05). 이 이하가 되면 종료.
    n_min : int
        W1: 최소 샘플 수 (기본 300).
    n_max : int
        W1: 최대 샘플 수 (기본 2000).

    Returns
    -------
    dict with keys:
        'rho'                : certified Lyapunov level (float)
        'rho_in'             : enter-LQR threshold = 0.5 * rho
        'rho_out'            : exit-to-swing-up threshold = 0.8 * rho
        'rho_max_empirical'  : max V0 over converged samples
        'n_converged'        : number of converging samples
        'n_total'            : total samples attempted
        'success_rate'       : n_converged / n_total
        'ci_width'           : W1: 최종 Wilson-score 95% CI 폭
    """
    # W1: cfg.actuator_saturation 에서 포화 한계 읽기
    u_sat = get_u_max(cfg)

    p = cfg.pack()
    max_angle = np.deg2rad(max_angle_deg)
    K_flat = np.ascontiguousarray(K.flatten(), dtype=np.float64)
    q_eq_c = np.ascontiguousarray(q_eq, dtype=np.float64)
    P_mat  = np.ascontiguousarray(P, dtype=np.float64)

    N_steps = int(t_horizon / dt)

    # ── Pre-generate Halton sequence for all n_max samples ──────────────
    # 8 dimensions: [cart, theta1, theta2, theta3, vx, vtheta1, vtheta2, vtheta3]
    halton_all = _halton_precompute(n_max, dim=8, seed=42)

    # Scale to physical ranges
    cart_range   = 0.5   # ±0.5 m
    angle_range  = max_angle
    vel_range    = 2.0   # ±2.0 rad/s or m/s

    # Convert [0,1] → centered ranges
    sq0_all  = (halton_all[:, 0] * 2 - 1) * cart_range
    sq1_all  = q_eq_c[1] + (halton_all[:, 1] * 2 - 1) * angle_range
    sq2_all  = q_eq_c[2] + (halton_all[:, 2] * 2 - 1) * angle_range
    sq3_all  = q_eq_c[3] + (halton_all[:, 3] * 2 - 1) * angle_range
    sdq0_all = (halton_all[:, 4] * 2 - 1) * vel_range
    sdq1_all = (halton_all[:, 5] * 2 - 1) * vel_range
    sdq2_all = (halton_all[:, 6] * 2 - 1) * vel_range
    sdq3_all = (halton_all[:, 7] * 2 - 1) * vel_range

    # Warmup _roa_batch_lyapunov JIT
    _roa_batch_lyapunov(
        2,
        sq0_all[:2], sq1_all[:2], sq2_all[:2], sq3_all[:2],
        sdq0_all[:2], sdq1_all[:2], sdq2_all[:2], sdq3_all[:2],
        q_eq_c, K_flat, P_mat, p, 5, dt, convergence_threshold, u_sat,
    )

    # ── Adaptive batch sampling with Wilson CI ──────────────────────────
    total_samples = 0
    all_converged = np.zeros(n_max, dtype=np.bool_)
    all_V0        = np.zeros(n_max, dtype=np.float64)

    batch_size = min(n_min, n_max)  # first batch = n_min
    while total_samples < n_max:
        end = min(total_samples + batch_size, n_max)
        n_batch = end - total_samples

        conv_batch, V0_batch = _roa_batch_lyapunov(
            n_batch,
            sq0_all[total_samples:end],
            sq1_all[total_samples:end],
            sq2_all[total_samples:end],
            sq3_all[total_samples:end],
            sdq0_all[total_samples:end],
            sdq1_all[total_samples:end],
            sdq2_all[total_samples:end],
            sdq3_all[total_samples:end],
            q_eq_c, K_flat, P_mat, p,
            N_steps, dt, convergence_threshold, u_sat,
        )
        all_converged[total_samples:end] = conv_batch
        all_V0[total_samples:end]        = V0_batch
        total_samples = end

        # Wilson CI check (after n_min samples)
        if total_samples >= n_min:
            n_conv = int(np.sum(all_converged[:total_samples]))
            ci = wilson_ci_width(n_conv, total_samples)
            if ci < target_ci_width:
                break

        batch_size = min(200, n_max - total_samples)  # subsequent batches = 200

    # Trim to actual samples used
    all_converged = all_converged[:total_samples]
    all_V0        = all_V0[:total_samples]

    # 최종 CI 폭
    n_converged = int(np.sum(all_converged))
    final_ci    = wilson_ci_width(n_converged, total_samples)

    if n_converged == 0:
        return {
            "rho": 1.0,
            "rho_in": 0.5,
            "rho_out": 0.8,
            "rho_max_empirical": 0.0,
            "n_converged": 0,
            "n_total": total_samples,
            "success_rate": 0.0,
            "ci_width": float(final_ci),
        }

    rho_max = float(np.max(all_V0[all_converged]))
    rho = safety_factor * rho_max
    rho_in  = 0.5 * rho
    rho_out = 0.8 * rho

    return {
        "rho": rho,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "rho_max_empirical": rho_max,
        "n_converged": n_converged,
        "n_total": total_samples,
        "success_rate": n_converged / total_samples,
        "ci_width": float(final_ci),
    }
