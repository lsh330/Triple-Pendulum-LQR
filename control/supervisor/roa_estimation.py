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
"""

import numpy as np
from numba import njit

from core.angle_utils import angle_wrap
from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast


@njit(cache=True)
def _roa_simulate_one(
    sq0_init, sq1_init, sq2_init, sq3_init,
    sdq0_init, sdq1_init, sdq2_init, sdq3_init,
    q_eq, K_flat, p, N_steps, dt, convergence_threshold,
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

        # Saturate at 200 N
        if u > 200.0:
            u = 200.0
        elif u < -200.0:
            u = -200.0

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
        Number of Monte Carlo samples.
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
    """
    p = cfg.pack()
    max_angle = np.deg2rad(max_angle_deg)
    K_flat = np.ascontiguousarray(K.flatten(), dtype=np.float64)
    q_eq_c = np.ascontiguousarray(q_eq, dtype=np.float64)

    rng = np.random.default_rng(42)
    lyap_values_converged = []
    N_steps = int(t_horizon / dt)

    for _ in range(n_samples):
        # Random initial perturbation around equilibrium
        dq = rng.uniform(-max_angle, max_angle, size=4)
        dq[0] = rng.uniform(-0.5, 0.5)   # Cart: smaller spatial range [m]
        ddq = rng.uniform(-2.0, 2.0, size=4)  # Initial velocities [rad/s or m/s]

        # Compute Lyapunov value at initial condition (Python side — runs once per sample)
        z0 = np.array([
            dq[0],
            angle_wrap(dq[1]),
            angle_wrap(dq[2]),
            angle_wrap(dq[3]),
            ddq[0], ddq[1], ddq[2], ddq[3],
        ])
        V0 = float(z0 @ P @ z0)

        # Initial scalar state
        sq0 = q_eq[0] + dq[0]
        sq1 = q_eq[1] + dq[1]
        sq2 = q_eq[2] + dq[2]
        sq3 = q_eq[3] + dq[3]
        sdq0, sdq1, sdq2, sdq3 = float(ddq[0]), float(ddq[1]), float(ddq[2]), float(ddq[3])

        # Run N_steps inside JIT kernel — zero Python interpreter overhead per RK4 step
        converged, _, _, _, _, _, _, _, _ = _roa_simulate_one(
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3,
            q_eq_c, K_flat, p, N_steps, dt, convergence_threshold,
        )

        if converged:
            lyap_values_converged.append(V0)

    if len(lyap_values_converged) == 0:
        # No convergence observed — return conservative fall-back values
        return {
            "rho": 1.0,
            "rho_in": 0.5,
            "rho_out": 0.8,
            "rho_max_empirical": 0.0,
            "n_converged": 0,
            "n_total": n_samples,
            "success_rate": 0.0,
        }

    rho_max = float(np.max(lyap_values_converged))
    rho = safety_factor * rho_max
    rho_in = 0.5 * rho
    rho_out = 0.8 * rho

    return {
        "rho": rho,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "rho_max_empirical": rho_max,
        "n_converged": len(lyap_values_converged),
        "n_total": n_samples,
        "success_rate": len(lyap_values_converged) / n_samples,
    }
