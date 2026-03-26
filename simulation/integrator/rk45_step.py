"""Dormand-Prince RK4(5) adaptive step integrator.

Provides embedded error estimation for automatic step size control.
Uses the standard Dormand-Prince coefficients (1980) with FSAL property.
"""

import numpy as np
from numba import njit
from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast


# Dormand-Prince coefficients
_A2 = 1.0/5.0
_A3 = 3.0/10.0
_A4 = 4.0/5.0
_A5 = 8.0/9.0

_B21 = 1.0/5.0
_B31 = 3.0/40.0; _B32 = 9.0/40.0
_B41 = 44.0/45.0; _B42 = -56.0/15.0; _B43 = 32.0/9.0
_B51 = 19372.0/6561.0; _B52 = -25360.0/2187.0; _B53 = 64448.0/6561.0; _B54 = -212.0/729.0
_B61 = 9017.0/3168.0; _B62 = -355.0/33.0; _B63 = 46732.0/5247.0; _B64 = 49.0/176.0; _B65 = -5103.0/18656.0

# 4th order weights (for error estimation)
_E1 = 71.0/57600.0; _E3 = -71.0/16695.0; _E4 = 71.0/1920.0; _E5 = -17253.0/339200.0; _E6 = 22.0/525.0; _E7 = -1.0/40.0

# 5th order weights (solution)
_D1 = 35.0/384.0; _D3 = 500.0/1113.0; _D4 = 125.0/192.0; _D5 = -2187.0/6784.0; _D6 = 11.0/84.0


@njit(cache=True)
def _eval_rhs(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p):
    """Evaluate RHS: returns (dq0..dq3, ddq0..ddq3)."""
    ddq0, ddq1, ddq2, ddq3 = forward_dynamics_fast(
        q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p)
    return dq0, dq1, dq2, dq3, ddq0, ddq1, ddq2, ddq3


@njit(cache=True)
def rk45_adaptive_step(q0, q1, q2, q3, dq0, dq1, dq2, dq3, u, p,
                       dt, atol=1e-8, rtol=1e-6):
    """Single adaptive RK45 step. Returns new state and actual step taken.

    Uses Dormand-Prince 4(5) with embedded error estimation.
    Returns (q0..q3, dq0..dq3, dt_actual, n_substeps).
    """
    t_remaining = dt
    n_substeps = 0
    n_attempts = 0
    h = dt

    sq0, sq1, sq2, sq3 = q0, q1, q2, q3
    sdq0, sdq1, sdq2, sdq3 = dq0, dq1, dq2, dq3

    while t_remaining > 1e-15:
        if h > t_remaining:
            h = t_remaining

        n_attempts += 1

        # Stage 1
        f1_q0, f1_q1, f1_q2, f1_q3, f1_d0, f1_d1, f1_d2, f1_d3 = _eval_rhs(
            sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, u, p)

        # Stage 2
        tq0 = sq0 + h*_B21*f1_q0; tq1 = sq1 + h*_B21*f1_q1
        tq2 = sq2 + h*_B21*f1_q2; tq3 = sq3 + h*_B21*f1_q3
        td0 = sdq0 + h*_B21*f1_d0; td1 = sdq1 + h*_B21*f1_d1
        td2 = sdq2 + h*_B21*f1_d2; td3 = sdq3 + h*_B21*f1_d3
        f2_q0, f2_q1, f2_q2, f2_q3, f2_d0, f2_d1, f2_d2, f2_d3 = _eval_rhs(
            tq0, tq1, tq2, tq3, td0, td1, td2, td3, u, p)

        # Stage 3
        tq0 = sq0 + h*(_B31*f1_q0 + _B32*f2_q0)
        tq1 = sq1 + h*(_B31*f1_q1 + _B32*f2_q1)
        tq2 = sq2 + h*(_B31*f1_q2 + _B32*f2_q2)
        tq3 = sq3 + h*(_B31*f1_q3 + _B32*f2_q3)
        td0 = sdq0 + h*(_B31*f1_d0 + _B32*f2_d0)
        td1 = sdq1 + h*(_B31*f1_d1 + _B32*f2_d1)
        td2 = sdq2 + h*(_B31*f1_d2 + _B32*f2_d2)
        td3 = sdq3 + h*(_B31*f1_d3 + _B32*f2_d3)
        f3_q0, f3_q1, f3_q2, f3_q3, f3_d0, f3_d1, f3_d2, f3_d3 = _eval_rhs(
            tq0, tq1, tq2, tq3, td0, td1, td2, td3, u, p)

        # Stage 4
        tq0 = sq0 + h*(_B41*f1_q0 + _B42*f2_q0 + _B43*f3_q0)
        tq1 = sq1 + h*(_B41*f1_q1 + _B42*f2_q1 + _B43*f3_q1)
        tq2 = sq2 + h*(_B41*f1_q2 + _B42*f2_q2 + _B43*f3_q2)
        tq3 = sq3 + h*(_B41*f1_q3 + _B42*f2_q3 + _B43*f3_q3)
        td0 = sdq0 + h*(_B41*f1_d0 + _B42*f2_d0 + _B43*f3_d0)
        td1 = sdq1 + h*(_B41*f1_d1 + _B42*f2_d1 + _B43*f3_d1)
        td2 = sdq2 + h*(_B41*f1_d2 + _B42*f2_d2 + _B43*f3_d2)
        td3 = sdq3 + h*(_B41*f1_d3 + _B42*f2_d3 + _B43*f3_d3)
        f4_q0, f4_q1, f4_q2, f4_q3, f4_d0, f4_d1, f4_d2, f4_d3 = _eval_rhs(
            tq0, tq1, tq2, tq3, td0, td1, td2, td3, u, p)

        # Stage 5
        tq0 = sq0 + h*(_B51*f1_q0 + _B52*f2_q0 + _B53*f3_q0 + _B54*f4_q0)
        tq1 = sq1 + h*(_B51*f1_q1 + _B52*f2_q1 + _B53*f3_q1 + _B54*f4_q1)
        tq2 = sq2 + h*(_B51*f1_q2 + _B52*f2_q2 + _B53*f3_q2 + _B54*f4_q2)
        tq3 = sq3 + h*(_B51*f1_q3 + _B52*f2_q3 + _B53*f3_q3 + _B54*f4_q3)
        td0 = sdq0 + h*(_B51*f1_d0 + _B52*f2_d0 + _B53*f3_d0 + _B54*f4_d0)
        td1 = sdq1 + h*(_B51*f1_d1 + _B52*f2_d1 + _B53*f3_d1 + _B54*f4_d1)
        td2 = sdq2 + h*(_B51*f1_d2 + _B52*f2_d2 + _B53*f3_d2 + _B54*f4_d2)
        td3 = sdq3 + h*(_B51*f1_d3 + _B52*f2_d3 + _B53*f3_d3 + _B54*f4_d3)
        f5_q0, f5_q1, f5_q2, f5_q3, f5_d0, f5_d1, f5_d2, f5_d3 = _eval_rhs(
            tq0, tq1, tq2, tq3, td0, td1, td2, td3, u, p)

        # Stage 6
        tq0 = sq0 + h*(_B61*f1_q0 + _B62*f2_q0 + _B63*f3_q0 + _B64*f4_q0 + _B65*f5_q0)
        tq1 = sq1 + h*(_B61*f1_q1 + _B62*f2_q1 + _B63*f3_q1 + _B64*f4_q1 + _B65*f5_q1)
        tq2 = sq2 + h*(_B61*f1_q2 + _B62*f2_q2 + _B63*f3_q2 + _B64*f4_q2 + _B65*f5_q2)
        tq3 = sq3 + h*(_B61*f1_q3 + _B62*f2_q3 + _B63*f3_q3 + _B64*f4_q3 + _B65*f5_q3)
        td0 = sdq0 + h*(_B61*f1_d0 + _B62*f2_d0 + _B63*f3_d0 + _B64*f4_d0 + _B65*f5_d0)
        td1 = sdq1 + h*(_B61*f1_d1 + _B62*f2_d1 + _B63*f3_d1 + _B64*f4_d1 + _B65*f5_d1)
        td2 = sdq2 + h*(_B61*f1_d2 + _B62*f2_d2 + _B63*f3_d2 + _B64*f4_d2 + _B65*f5_d2)
        td3 = sdq3 + h*(_B61*f1_d3 + _B62*f2_d3 + _B63*f3_d3 + _B64*f4_d3 + _B65*f5_d3)
        f6_q0, f6_q1, f6_q2, f6_q3, f6_d0, f6_d1, f6_d2, f6_d3 = _eval_rhs(
            tq0, tq1, tq2, tq3, td0, td1, td2, td3, u, p)

        # 5th order solution
        nq0 = sq0 + h*(_D1*f1_q0 + _D3*f3_q0 + _D4*f4_q0 + _D5*f5_q0 + _D6*f6_q0)
        nq1 = sq1 + h*(_D1*f1_q1 + _D3*f3_q1 + _D4*f4_q1 + _D5*f5_q1 + _D6*f6_q1)
        nq2 = sq2 + h*(_D1*f1_q2 + _D3*f3_q2 + _D4*f4_q2 + _D5*f5_q2 + _D6*f6_q2)
        nq3 = sq3 + h*(_D1*f1_q3 + _D3*f3_q3 + _D4*f4_q3 + _D5*f5_q3 + _D6*f6_q3)
        nd0 = sdq0 + h*(_D1*f1_d0 + _D3*f3_d0 + _D4*f4_d0 + _D5*f5_d0 + _D6*f6_d0)
        nd1 = sdq1 + h*(_D1*f1_d1 + _D3*f3_d1 + _D4*f4_d1 + _D5*f5_d1 + _D6*f6_d1)
        nd2 = sdq2 + h*(_D1*f1_d2 + _D3*f3_d2 + _D4*f4_d2 + _D5*f5_d2 + _D6*f6_d2)
        nd3 = sdq3 + h*(_D1*f1_d3 + _D3*f3_d3 + _D4*f4_d3 + _D5*f5_d3 + _D6*f6_d3)

        # Stage 7 for error estimation
        f7_q0, f7_q1, f7_q2, f7_q3, f7_d0, f7_d1, f7_d2, f7_d3 = _eval_rhs(
            nq0, nq1, nq2, nq3, nd0, nd1, nd2, nd3, u, p)

        # Error estimation (difference between 4th and 5th order)
        err_max = 0.0
        for val, sc in [
            (h*(_E1*f1_q0 + _E3*f3_q0 + _E4*f4_q0 + _E5*f5_q0 + _E6*f6_q0 + _E7*f7_q0), atol + rtol*max(abs(sq0), abs(nq0))),
            (h*(_E1*f1_q1 + _E3*f3_q1 + _E4*f4_q1 + _E5*f5_q1 + _E6*f6_q1 + _E7*f7_q1), atol + rtol*max(abs(sq1), abs(nq1))),
            (h*(_E1*f1_q2 + _E3*f3_q2 + _E4*f4_q2 + _E5*f5_q2 + _E6*f6_q2 + _E7*f7_q2), atol + rtol*max(abs(sq2), abs(nq2))),
            (h*(_E1*f1_q3 + _E3*f3_q3 + _E4*f4_q3 + _E5*f5_q3 + _E6*f6_q3 + _E7*f7_q3), atol + rtol*max(abs(sq3), abs(nq3))),
            (h*(_E1*f1_d0 + _E3*f3_d0 + _E4*f4_d0 + _E5*f5_d0 + _E6*f6_d0 + _E7*f7_d0), atol + rtol*max(abs(sdq0), abs(nd0))),
            (h*(_E1*f1_d1 + _E3*f3_d1 + _E4*f4_d1 + _E5*f5_d1 + _E6*f6_d1 + _E7*f7_d1), atol + rtol*max(abs(sdq1), abs(nd1))),
            (h*(_E1*f1_d2 + _E3*f3_d2 + _E4*f4_d2 + _E5*f5_d2 + _E6*f6_d2 + _E7*f7_d2), atol + rtol*max(abs(sdq2), abs(nd2))),
            (h*(_E1*f1_d3 + _E3*f3_d3 + _E4*f4_d3 + _E5*f5_d3 + _E6*f6_d3 + _E7*f7_d3), atol + rtol*max(abs(sdq3), abs(nd3))),
        ]:
            e = abs(val) / sc
            if e > err_max:
                err_max = e

        if err_max <= 1.0:
            # Accept step
            sq0, sq1, sq2, sq3 = nq0, nq1, nq2, nq3
            sdq0, sdq1, sdq2, sdq3 = nd0, nd1, nd2, nd3
            t_remaining -= h
            n_substeps += 1

        # Adjust step size (safety factor 0.9)
        if err_max > 0.0:
            h = h * min(5.0, max(0.2, 0.9 * err_max**(-0.2)))
        else:
            h = h * 5.0

        # Prevent infinite loops
        if n_substeps > 10000 or n_attempts > 50000:
            break

    return sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3, n_substeps
