"""Zero-allocation switching simulation loop for form transitions.

FSM Modes
---------
  MODE_SWING_UP (0)   : energy-based controller  u = k_e * (E* - E) * dx
  MODE_LQR_CATCH (1)  : LQR u = -K*z, with Lyapunov hysteresis monitoring
  MODE_STABILIZED (2) : LQR hold at current target; advance stage after
                        STABILIZE_STEPS consecutive steps below threshold

Stage advance logic
-------------------
  After MODE_STABILIZED fires, current_stage increments.
  If current_stage < n_stages the mode resets to MODE_SWING_UP for the next
  target.  Once current_stage == n_stages all transitions are complete and
  the loop simply holds the final LQR.
"""

import numpy as np
from numba import njit

from dynamics.forward_dynamics.forward_dynamics_fast import rk4_step_fast  # noqa: F401
from core.angle_utils import angle_wrap                                      # noqa: F401
from control.swing_up.energy_computation import total_energy_scalar         # noqa: F401

# Mode integer constants (used both here and in runner/analysis)
MODE_SWING_UP = 0
MODE_LQR_CATCH = 1
MODE_STABILIZED = 2


@njit(cache=True, fastmath=True, boundscheck=False)
def _lyapunov_value(sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, q_eq, P_flat):
    """Compute scalar Lyapunov value V = z^T P z.

    Parameters
    ----------
    sq0..sq3 : float
        Current configuration [x, theta1, theta2, theta3].
    sdq0..sdq3 : float
        Current velocities.
    q_eq : 1-D array (4,)
        Equilibrium configuration for this stage.
    P_flat : 2-D array (8, 8)
        Lyapunov / Riccati matrix.

    Returns
    -------
    float
        V = z^T P z  (always >= 0 for PD P).

    Implementation note
    -------------------
    Fully inlined scalar expansion with symmetry exploitation.
    P is symmetric, so V = sum_i P[i,i]*zi^2 + 2*sum_{i<j} P[i,j]*zi*zj.
    This eliminates the np.empty(8) allocation and the 8x8=64-iteration
    double loop, replacing them with 8 diagonal + 28 off-diagonal = 36 ops.
    """
    # State error vector as 8 scalars — no heap allocation
    z0 = sq0 - q_eq[0]
    z1 = angle_wrap(sq1 - q_eq[1])
    z2 = angle_wrap(sq2 - q_eq[2])
    z3 = angle_wrap(sq3 - q_eq[3])
    z4 = sdq0
    z5 = sdq1
    z6 = sdq2
    z7 = sdq3

    # Diagonal terms: P[i,i] * zi^2
    V = (P_flat[0, 0] * z0 * z0
       + P_flat[1, 1] * z1 * z1
       + P_flat[2, 2] * z2 * z2
       + P_flat[3, 3] * z3 * z3
       + P_flat[4, 4] * z4 * z4
       + P_flat[5, 5] * z5 * z5
       + P_flat[6, 6] * z6 * z6
       + P_flat[7, 7] * z7 * z7)

    # Off-diagonal terms (upper triangle): 2 * P[i,j] * zi * zj
    # Row 0 (7 terms)
    V += 2.0 * (P_flat[0, 1] * z0 * z1
              + P_flat[0, 2] * z0 * z2
              + P_flat[0, 3] * z0 * z3
              + P_flat[0, 4] * z0 * z4
              + P_flat[0, 5] * z0 * z5
              + P_flat[0, 6] * z0 * z6
              + P_flat[0, 7] * z0 * z7
    # Row 1 (6 terms)
              + P_flat[1, 2] * z1 * z2
              + P_flat[1, 3] * z1 * z3
              + P_flat[1, 4] * z1 * z4
              + P_flat[1, 5] * z1 * z5
              + P_flat[1, 6] * z1 * z6
              + P_flat[1, 7] * z1 * z7
    # Row 2 (5 terms)
              + P_flat[2, 3] * z2 * z3
              + P_flat[2, 4] * z2 * z4
              + P_flat[2, 5] * z2 * z5
              + P_flat[2, 6] * z2 * z6
              + P_flat[2, 7] * z2 * z7
    # Row 3 (4 terms)
              + P_flat[3, 4] * z3 * z4
              + P_flat[3, 5] * z3 * z5
              + P_flat[3, 6] * z3 * z6
              + P_flat[3, 7] * z3 * z7
    # Row 4 (3 terms)
              + P_flat[4, 5] * z4 * z5
              + P_flat[4, 6] * z4 * z6
              + P_flat[4, 7] * z4 * z7
    # Row 5 (2 terms)
              + P_flat[5, 6] * z5 * z6
              + P_flat[5, 7] * z5 * z7
    # Row 6 (1 term)
              + P_flat[6, 7] * z6 * z7)
    return V


@njit(cache=True, fastmath=True, boundscheck=False)
def _run_loop_switching(
    N, dt,
    sq0_init, sq1_init, sq2_init, sq3_init,
    sdq0_init, sdq1_init, sdq2_init, sdq3_init,
    p,
    n_stages,
    all_q_eq,       # (n_stages, 4)
    all_K_flat,     # (n_stages, 8)
    all_P_flat,     # (n_stages, 8, 8)
    all_E_target,   # (n_stages,)
    all_rho_in,     # (n_stages,)
    all_rho_out,    # (n_stages,)
    k_energy,
    u_max,
):
    """Zero-allocation switching simulation loop.

    Parameters
    ----------
    N : int
        Total number of time steps.
    dt : float
        Integration timestep (s).
    sq0_init .. sdq3_init : float
        Initial scalar state.
    p : 1-D array (13,)
        Packed parameter vector.
    n_stages : int
        Number of transitions (= len(transition_path) - 1).
    all_q_eq : (n_stages, 4)  float64
        Equilibrium configurations for each stage target.
    all_K_flat : (n_stages, 8)  float64
        Flattened LQR gain vectors.
    all_P_flat : (n_stages, 8, 8)  float64
        Lyapunov matrices.
    all_E_target : (n_stages,)  float64
        Target physical energies for swing-up.
    all_rho_in : (n_stages,)  float64
        Lyapunov level threshold for entering LQR catch.
    all_rho_out : (n_stages,)  float64
        Lyapunov level threshold for exiting back to swing-up.
    k_energy : float
        Energy shaping gain.
    u_max : float
        Actuator saturation limit (N).

    Returns
    -------
    q_arr : (N, 4)  float64
    dq_arr : (N, 4)  float64
    u_arr : (N,)  float64
    mode_arr : (N,)  int32   -- 0=SWING_UP, 1=LQR_CATCH, 2=STABILIZED, -1=NaN
    stage_arr : (N,)  int32  -- current stage index (-1 after NaN)
    energy_arr : (N,)  float64
    """
    # Output arrays
    q_arr = np.empty((N, 4))
    dq_arr = np.empty((N, 4))
    u_arr = np.empty(N)
    mode_arr = np.empty(N, dtype=np.int32)
    stage_arr = np.empty(N, dtype=np.int32)
    energy_arr = np.empty(N)

    # State scalars
    sq0 = sq0_init;  sq1 = sq1_init;  sq2 = sq2_init;  sq3 = sq3_init
    sdq0 = sdq0_init; sdq1 = sdq1_init; sdq2 = sdq2_init; sdq3 = sdq3_init

    # FSM state
    current_stage = 0
    current_mode = MODE_SWING_UP
    stabilize_count = 0
    # 0.5 s at dt=0.001 -> 500 consecutive steps below threshold
    STABILIZE_STEPS = 500

    for k in range(N):
        # ----------------------------------------------------------------
        # Record current state
        # ----------------------------------------------------------------
        q_arr[k, 0] = sq0;  q_arr[k, 1] = sq1
        q_arr[k, 2] = sq2;  q_arr[k, 3] = sq3
        dq_arr[k, 0] = sdq0; dq_arr[k, 1] = sdq1
        dq_arr[k, 2] = sdq2; dq_arr[k, 3] = sdq3
        stage_arr[k] = current_stage
        mode_arr[k] = current_mode

        E_current = total_energy_scalar(
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, p
        )
        energy_arr[k] = E_current

        # ----------------------------------------------------------------
        # Control law
        # ----------------------------------------------------------------
        if current_stage >= n_stages:
            # All transitions complete — hold final LQR
            stage_idx = n_stages - 1
            q_eq = all_q_eq[stage_idx]
            K_flat = all_K_flat[stage_idx]
            z0 = sq0 - q_eq[0]
            z1 = angle_wrap(sq1 - q_eq[1])
            z2 = angle_wrap(sq2 - q_eq[2])
            z3 = angle_wrap(sq3 - q_eq[3])
            u = -(K_flat[0] * z0 + K_flat[1] * z1 + K_flat[2] * z2
                  + K_flat[3] * z3 + K_flat[4] * sdq0 + K_flat[5] * sdq1
                  + K_flat[6] * sdq2 + K_flat[7] * sdq3)
            mode_arr[k] = MODE_STABILIZED

        else:
            q_eq = all_q_eq[current_stage]
            K_flat = all_K_flat[current_stage]
            P_flat = all_P_flat[current_stage]
            E_target = all_E_target[current_stage]
            rho_in = all_rho_in[current_stage]
            rho_out = all_rho_out[current_stage]

            V_lyap = _lyapunov_value(
                sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3,
                q_eq, P_flat
            )

            if current_mode == MODE_SWING_UP:
                # Energy-based swing-up: u = k_e * (E* - E) * dx_cart
                u = k_energy * (E_target - E_current) * sdq0

                # Enter LQR catch when state enters ROA
                if V_lyap < rho_in:
                    current_mode = MODE_LQR_CATCH

            elif current_mode == MODE_LQR_CATCH:
                z0 = sq0 - q_eq[0]
                z1 = angle_wrap(sq1 - q_eq[1])
                z2 = angle_wrap(sq2 - q_eq[2])
                z3 = angle_wrap(sq3 - q_eq[3])
                u = -(K_flat[0] * z0 + K_flat[1] * z1 + K_flat[2] * z2
                      + K_flat[3] * z3 + K_flat[4] * sdq0 + K_flat[5] * sdq1
                      + K_flat[6] * sdq2 + K_flat[7] * sdq3)

                # Hysteresis: exit back to swing-up if state leaves ROA
                if V_lyap > rho_out:
                    current_mode = MODE_SWING_UP
                    stabilize_count = 0
                else:
                    # Check for sustained convergence
                    dev = (abs(z0) + abs(z1) + abs(z2) + abs(z3)
                           + abs(sdq0) + abs(sdq1) + abs(sdq2) + abs(sdq3))
                    if dev < 0.1:
                        stabilize_count += 1
                    else:
                        stabilize_count = 0

                    if stabilize_count >= STABILIZE_STEPS:
                        current_mode = MODE_STABILIZED

            else:  # MODE_STABILIZED
                z0 = sq0 - q_eq[0]
                z1 = angle_wrap(sq1 - q_eq[1])
                z2 = angle_wrap(sq2 - q_eq[2])
                z3 = angle_wrap(sq3 - q_eq[3])
                u = -(K_flat[0] * z0 + K_flat[1] * z1 + K_flat[2] * z2
                      + K_flat[3] * z3 + K_flat[4] * sdq0 + K_flat[5] * sdq1
                      + K_flat[6] * sdq2 + K_flat[7] * sdq3)

                # Advance to next stage
                current_stage += 1
                if current_stage < n_stages:
                    current_mode = MODE_SWING_UP
                    stabilize_count = 0

        # ----------------------------------------------------------------
        # Actuator saturation
        # ----------------------------------------------------------------
        if u > u_max:
            u = u_max
        elif u < -u_max:
            u = -u_max
        u_arr[k] = u

        # ----------------------------------------------------------------
        # RK4 integration (skip on last step)
        # ----------------------------------------------------------------
        if k < N - 1:
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3 = rk4_step_fast(
                sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u, p, dt
            )

            # NaN divergence check — fill remainder and break
            if sq0 != sq0 or sq1 != sq1 or sq2 != sq2 or sq3 != sq3:
                for fill_k in range(k + 1, N):
                    q_arr[fill_k, 0] = np.nan; q_arr[fill_k, 1] = np.nan
                    q_arr[fill_k, 2] = np.nan; q_arr[fill_k, 3] = np.nan
                    dq_arr[fill_k, 0] = np.nan; dq_arr[fill_k, 1] = np.nan
                    dq_arr[fill_k, 2] = np.nan; dq_arr[fill_k, 3] = np.nan
                    u_arr[fill_k] = np.nan
                    mode_arr[fill_k] = -1
                    stage_arr[fill_k] = -1
                    energy_arr[fill_k] = np.nan
                break

    return q_arr, dq_arr, u_arr, mode_arr, stage_arr, energy_arr
