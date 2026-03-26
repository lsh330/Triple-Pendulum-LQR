#!/usr/bin/env python
"""Pre-compile all Numba JIT functions to eliminate first-run latency.

Run once after installation or after clearing __pycache__:
    python prebuild_cache.py

This triggers compilation of all @njit(cache=True) functions and stores
the compiled artifacts in __pycache__/ directories. Subsequent imports
will load from cache instead of recompiling.
"""

import sys
import time


def prebuild():
    """Trigger JIT compilation of all cached functions."""
    print("Pre-building Numba JIT cache...")
    print(f"Python {sys.version}")

    stages = []
    t_total = time.perf_counter()

    # Stage 1: Core dynamics
    t0 = time.perf_counter()
    from simulation.warmup import warmup_jit
    warmup_jit()
    stages.append(("Core dynamics + simulation loops", time.perf_counter() - t0))

    # Stage 2: Linearization JIT
    t0 = time.perf_counter()
    from parameters.config import SystemConfig
    import numpy as np
    cfg = SystemConfig(mc=1, m1=1, m2=1, m3=1, L1=1, L2=1, L3=1)
    from control.linearization.jit_jacobians import compute_jacobians_jit
    q_eq = cfg.equilibrium
    p = cfg.pack()
    compute_jacobians_jit(q_eq, np.zeros(4), 0.0, p)
    stages.append(("Linearization Jacobians", time.perf_counter() - t0))

    # Stage 3: ROA parallel kernel
    t0 = time.perf_counter()
    from analysis.region_of_attraction import _roa_batch
    K_flat = np.zeros(8)
    t1 = np.array([0.01, -0.01])
    t2 = np.array([0.0, 0.0])
    t3 = np.array([0.0, 0.0])
    _roa_batch(2, 10, 0.001, q_eq, K_flat, p, t1, t2, t3, 0.017)
    stages.append(("ROA parallel kernel", time.perf_counter() - t0))

    # Stage 4: RK45 adaptive integrator
    t0 = time.perf_counter()
    try:
        from simulation.integrator.rk45_step import rk45_adaptive_step
        rk45_adaptive_step(0.0, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p, 0.001)
        stages.append(("RK45 adaptive integrator", time.perf_counter() - t0))
    except ImportError:
        stages.append(("RK45 adaptive integrator (skipped)", 0.0))

    t_elapsed = time.perf_counter() - t_total

    print("\n" + "=" * 50)
    print("  JIT Cache Pre-build Summary")
    print("=" * 50)
    for name, dt in stages:
        print(f"  {name:40s} {dt:.2f}s")
    print("-" * 50)
    print(f"  {'Total':40s} {t_elapsed:.2f}s")
    print("=" * 50)
    print("\nCache files stored in __pycache__/ directories.")
    print("Subsequent runs will skip compilation.")


if __name__ == "__main__":
    prebuild()
