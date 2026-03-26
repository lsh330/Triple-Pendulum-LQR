"""Pre-trigger JIT compilation of all numba functions at import time."""
import numpy as np


def warmup_jit():
    """Call all @njit functions once with dummy data to trigger compilation."""
    from parameters.config import SystemConfig
    cfg = SystemConfig(mc=1, m1=1, m2=1, m3=1, L1=1, L2=1, L3=1)
    p = cfg.pack()
    q = cfg.equilibrium
    dq = np.zeros(4)

    # Trigger array-based dynamics (needed for linearization Jacobians)
    from dynamics.forward_dynamics.forward_dynamics import forward_dynamics
    forward_dynamics(q, dq, 0.0, p)

    # Trigger fast scalar dynamics and RK4 (used by main simulation + ROA)
    from dynamics.forward_dynamics.forward_dynamics_fast import forward_dynamics_fast, rk4_step_fast
    forward_dynamics_fast(q[0], q[1], q[2], q[3], 0.0, 0.0, 0.0, 0.0, 0.0, p)
    rk4_step_fast(q[0], q[1], q[2], q[3], 0.0, 0.0, 0.0, 0.0, 0.0, p, 0.001)

    # Trigger fast simulation loops (fixed-gain and gain-scheduled)
    dev = np.array([-0.35, 0.0, 0.35])
    Ks = np.zeros((3, 8))
    from simulation.loop.time_loop_fast import _run_loop_fast, _run_loop_gs_fast
    _run_loop_fast(3, 0.001, q, dq, q, np.zeros(8), p, np.empty(0), 1e30)
    _run_loop_gs_fast(3, 0.001, q, dq, q, p, np.empty(0), dev, Ks, 1e30)

    # Trigger ROA fast kernel (parallel scalar dynamics)
    from analysis.region_of_attraction import _roa_batch
    t1 = np.array([0.01, -0.01])
    t2 = np.array([0.0, 0.0])
    t3 = np.array([0.0, 0.0])
    _roa_batch(2, 5, 0.001, q, np.zeros(8), p, t1, t2, t3, 0.017)
