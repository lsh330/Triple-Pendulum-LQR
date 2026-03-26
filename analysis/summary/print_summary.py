"""Print a concise simulation summary to the console."""

import numpy as np


def print_summary(q, dq, state, u_ctrl, u_dist, freq_data) -> None:
    """Print key simulation metrics."""
    print("=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)

    # State ranges
    print(f"\n  Cart position : [{q[:, 0].min():+.4f}, {q[:, 0].max():+.4f}] m")
    print(f"  Cart velocity : [{dq[:, 0].min():+.4f}, {dq[:, 0].max():+.4f}] m/s")

    for i, key in enumerate(["dth1", "dth2", "dth3"], 1):
        arr = state[key]
        print(f"  dtheta{i}       : [{np.degrees(arr.min()):+.4f}, "
              f"{np.degrees(arr.max()):+.4f}] deg")

    # Control effort
    print(f"\n  Control force  : [{u_ctrl.min():+.2f}, {u_ctrl.max():+.2f}] N")
    print(f"  Disturbance    : [{u_dist.min():+.2f}, {u_dist.max():+.2f}] N")

    # Frequency-domain data (if available)
    if freq_data is not None:
        pm = freq_data.get("phase_margin", np.nan)
        gm = freq_data.get("gain_margin_dB", np.nan)
        os_pct = freq_data.get("overshoot", np.nan)
        ts = freq_data.get("t_settle", np.nan)
        Ms = freq_data.get("Ms", np.nan)
        Mt = freq_data.get("Mt", np.nan)
        print(f"\n  Phase margin   : {pm:.2f} deg")
        print(f"  Gain margin    : {gm:.2f} dB")
        print(f"  Overshoot      : {os_pct:.2f} %")
        print(f"  Settling time  : {ts:.4f} s")
        print(f"  Ms (peak |S|)  : {Ms:.3f}")
        print(f"  Mt (peak |T|)  : {Mt:.3f}")

    print("=" * 60)
