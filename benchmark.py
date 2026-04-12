"""Performance benchmark for the triple pendulum simulation.

Measures
--------
1. _run_loop_fast         : 15 s simulation (warmup excluded)
2. _run_loop_switching    : 30 s simulation (warmup excluded), JIT compile time reported
3. total_energy_scalar    : 1 000 000 calls (warmup excluded)
4. estimate_lyapunov_roa  : single call timing (JIT warmup excluded)

Run from the project root:
    python benchmark.py
"""

import sys
import os
import time

import numpy as np

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hline():
    print("-" * 60)


def _fmt_hz(n_steps, elapsed_s):
    """Return formatted Hz string."""
    if elapsed_s <= 0:
        return "N/A"
    hz = n_steps / elapsed_s
    if hz >= 1e6:
        return f"{hz/1e6:.2f} MHz  ({hz:.3e} steps/s)"
    elif hz >= 1e3:
        return f"{hz/1e3:.1f} kHz  ({hz:.3e} steps/s)"
    return f"{hz:.1f} Hz  ({hz:.3e} steps/s)"


# ---------------------------------------------------------------------------
# 0. Setup: system parameters
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  Triple Pendulum Simulation - Performance Benchmark")
print("=" * 60)
print()
print("Building system configuration (Medrano-Cerda params) ...")

from parameters.config import SystemConfig
from parameters.equilibrium import equilibrium

cfg = SystemConfig(mc=3.0, m1=1.323, m2=1.389, m3=0.8655,
                   L1=0.402, L2=0.332, L3=0.720)
p = cfg.pack()
q_eq = equilibrium("UUU")    # Fully upright equilibrium
dt = 0.001

print(f"  p shape: {p.shape},  q_eq: {q_eq}")
print()

# ---------------------------------------------------------------------------
# 1. Benchmark: _run_loop_fast  (15 s simulation)
# ---------------------------------------------------------------------------

from simulation.loop.time_loop_fast import _run_loop_fast
from control.lqr import compute_lqr_gains

K, A, B, P, Q, R = compute_lqr_gains(cfg)
K_flat = K.flatten().astype(np.float64)

T_fast_s = 15.0
N_fast = int(T_fast_s / dt)
dq0 = np.zeros(4)
q0_fast = q_eq.copy()
q0_fast[1] += 0.05   # small perturbation

print("--- Benchmark 1: _run_loop_fast (15 s simulation) ---")
print("  [warmup] first call -JIT compilation ...")
t_wup0 = time.perf_counter()
_run_loop_fast(10, dt, q0_fast, dq0, q_eq, K_flat, p, np.empty(0), 200.0)
t_wup1 = time.perf_counter()
print(f"  Warmup (JIT compile) took: {t_wup1 - t_wup0:.3f} s")

print(f"  Running {N_fast} steps ({T_fast_s} s real time) ...")
t0 = time.perf_counter()
res_fast = _run_loop_fast(N_fast, dt, q0_fast, dq0, q_eq, K_flat, p, np.empty(0), 200.0)
t1 = time.perf_counter()
elapsed_fast = t1 - t0
rt_ratio_fast = T_fast_s / elapsed_fast

print(f"  Elapsed : {elapsed_fast*1000:.2f} ms")
print(f"  Rate    : {_fmt_hz(N_fast, elapsed_fast)}")
print(f"  Real-time ratio : {rt_ratio_fast:.0f}x  (simulation/wall-clock)")
_hline()
print()

# ---------------------------------------------------------------------------
# 2. Benchmark: _run_loop_switching  (30 s simulation)
# ---------------------------------------------------------------------------

print("--- Benchmark 2: _run_loop_switching (30 s simulation) ---")
print("  Building FormSwitchSupervisor ...")
print("  (includes LQR solve + ROA estimation -this is setup, not benchmarked)")

t_setup0 = time.perf_counter()
from control.supervisor.form_switch_supervisor import FormSwitchSupervisor
sup = FormSwitchSupervisor(cfg, k_energy=50.0, u_max=200.0)
path = ["DDD", "UUU"]
packed = sup.pack_for_njit(path)
t_setup1 = time.perf_counter()
print(f"  Supervisor setup: {t_setup1 - t_setup0:.2f} s")

from simulation.loop.time_loop_switching import _run_loop_switching

n_stages   = packed["n_stages"]
all_q_eq   = packed["all_q_eq"]
all_K_flat = packed["all_K_flat"]
all_P_flat = packed["all_P_flat"]
all_E_tgt  = packed["all_E_target"]
all_rho_in = packed["all_rho_in"]
all_rho_out= packed["all_rho_out"]
k_energy   = packed["k_energy"]
u_max      = packed["u_max"]

# Start from DDD (hanging down) with small perturbation
q0_sw = equilibrium("DDD").copy()
q0_sw[1] += 0.02

T_sw_s = 30.0
N_sw = int(T_sw_s / dt)

def _call_switching(N):
    return _run_loop_switching(
        N, dt,
        float(q0_sw[0]), float(q0_sw[1]), float(q0_sw[2]), float(q0_sw[3]),
        0.0, 0.0, 0.0, 0.0,
        p, n_stages,
        all_q_eq, all_K_flat, all_P_flat,
        all_E_tgt, all_rho_in, all_rho_out,
        k_energy, u_max,
    )

print("  [warmup] first call -JIT compilation ...")
t_wup0 = time.perf_counter()
_call_switching(10)
t_wup1 = time.perf_counter()
t_jit_sw = t_wup1 - t_wup0
print(f"  Warmup (JIT compile) took: {t_jit_sw:.3f} s")

print(f"  Running {N_sw} steps ({T_sw_s} s real time) ...")
t0 = time.perf_counter()
res_sw = _call_switching(N_sw)
t1 = time.perf_counter()
elapsed_sw = t1 - t0
rt_ratio_sw = T_sw_s / elapsed_sw

print(f"  Elapsed : {elapsed_sw*1000:.2f} ms")
print(f"  Rate    : {_fmt_hz(N_sw, elapsed_sw)}")
print(f"  Real-time ratio : {rt_ratio_sw:.0f}x  (simulation/wall-clock)")

# Check for NaN divergence
q_arr_sw = res_sw[0]
nan_mask = np.isnan(q_arr_sw[:, 0])
first_nan = int(np.argmax(nan_mask)) if nan_mask.any() else -1
if first_nan > 0:
    print(f"  [info] Diverged at step {first_nan} ({first_nan*dt:.2f} s) -NaN fill rest")
else:
    print(f"  [info] No divergence in {T_sw_s} s")
_hline()
print()

# ---------------------------------------------------------------------------
# 3. Benchmark: total_energy_scalar  (1 000 000 calls)
# ---------------------------------------------------------------------------

print("--- Benchmark 3: total_energy_scalar (1 000 000 calls) ---")

from control.swing_up.energy_computation import total_energy_scalar

# Warmup (trigger JIT -likely already compiled, but ensure)
_ = total_energy_scalar(0.0, 0.1, 0.05, 0.02, 0.3, -0.1, 0.2, -0.2, p)

N_ENERGY = 1_000_000
sq0_b = 0.0; sq1_b = 0.52; sq2_b = 0.30; sq3_b = 0.10
sdq0_b = 0.5; sdq1_b = -0.2; sdq2_b = 0.1; sdq3_b = 0.0

print(f"  Running {N_ENERGY:,} calls ...")
t0 = time.perf_counter()
for _ in range(N_ENERGY):
    _ = total_energy_scalar(sq0_b, sq1_b, sq2_b, sq3_b,
                            sdq0_b, sdq1_b, sdq2_b, sdq3_b, p)
t1 = time.perf_counter()
elapsed_energy = t1 - t0
ns_per_call = elapsed_energy * 1e9 / N_ENERGY

print(f"  Total elapsed : {elapsed_energy*1000:.2f} ms")
print(f"  Per call      : {ns_per_call:.1f} ns")
print(f"  Throughput    : {_fmt_hz(N_ENERGY, elapsed_energy)}")
_hline()
print()

# ---------------------------------------------------------------------------
# 4. Benchmark: estimate_lyapunov_roa  (single call, JIT warmup excluded)
# ---------------------------------------------------------------------------

print("--- Benchmark 4: estimate_lyapunov_roa (n_samples=300, t_horizon=3 s) ---")
print("  [warmup] triggering _roa_simulate_one JIT compilation ...")

from control.supervisor.roa_estimation import _roa_simulate_one, estimate_lyapunov_roa

t_wup0 = time.perf_counter()
_roa_simulate_one(
    float(q_eq[0]), float(q_eq[1]) + 0.1, float(q_eq[2]), float(q_eq[3]),
    0.1, 0.0, 0.0, 0.0,
    np.ascontiguousarray(q_eq, dtype=np.float64),
    K_flat,
    p,
    5,   # tiny N_steps for warmup
    dt,
    0.1,
)
t_wup1 = time.perf_counter()
print(f"  Warmup (JIT compile) took: {t_wup1 - t_wup0:.3f} s")

print("  Running estimate_lyapunov_roa (n_samples=300) ...")
t0 = time.perf_counter()
roa_result = estimate_lyapunov_roa(
    cfg, K, P, q_eq,
    n_samples=300,
    max_angle_deg=25.0,
    t_horizon=3.0,
    dt=dt,
)
t1 = time.perf_counter()
elapsed_roa = t1 - t0

print(f"  Elapsed       : {elapsed_roa:.3f} s")
print(f"  Per sample    : {elapsed_roa/300*1000:.2f} ms")
print(f"  rho           : {roa_result['rho']:.6f}")
print(f"  rho_in        : {roa_result['rho_in']:.6f}")
print(f"  rho_out       : {roa_result['rho_out']:.6f}")
print(f"  success_rate  : {roa_result['success_rate']*100:.1f}%  "
      f"({roa_result['n_converged']}/{roa_result['n_total']})")
_hline()
print()

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  _run_loop_fast       (15 s): {rt_ratio_fast:>8.0f}x real-time"
      f"   [{elapsed_fast*1000:.1f} ms wall]")
print(f"  _run_loop_switching  (30 s): {rt_ratio_sw:>8.0f}x real-time"
      f"   [{elapsed_sw*1000:.1f} ms wall]")
print(f"  total_energy_scalar (1M):    {ns_per_call:>6.1f} ns/call"
      f"   [{elapsed_energy*1000:.1f} ms total]")
print(f"  estimate_lyapunov_roa:       {elapsed_roa:.3f} s"
      f"   [{elapsed_roa/300*1000:.2f} ms/sample]")
overhead_frac = (elapsed_sw - elapsed_fast * (T_sw_s / T_fast_s)) / elapsed_sw
print()
print(f"  Switching vs fast loop overhead: {overhead_frac*100:+.1f}%")
print("=" * 60)
print()
