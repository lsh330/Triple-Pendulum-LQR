"""Print a concise simulation summary to the console."""

import numpy as np

from utils.logger import get_logger

log = get_logger()


def print_summary(q, dq, state, u_ctrl, u_dist, freq_data, u_max=None) -> None:
    """Print key simulation metrics."""
    log.info("=" * 60)
    log.info("  SIMULATION SUMMARY")
    log.info("=" * 60)

    # State ranges
    log.info("  Cart position : [%+.4f, %+.4f] m", q[:, 0].min(), q[:, 0].max())
    log.info("  Cart velocity : [%+.4f, %+.4f] m/s", dq[:, 0].min(), dq[:, 0].max())

    for i, key in enumerate(["dth1", "dth2", "dth3"], 1):
        arr = state[key]
        log.info("  dtheta%d       : [%+.4f, %+.4f] deg",
                 i, np.degrees(arr.min()), np.degrees(arr.max()))

    # Control effort
    log.info("  Control force  : [%+.2f, %+.2f] N", u_ctrl.min(), u_ctrl.max())
    # Saturation statistics
    peak_u = max(abs(u_ctrl.min()), abs(u_ctrl.max()))
    if u_max is not None and u_max < 1e20:
        n_saturated = int(np.sum(np.abs(u_ctrl) >= u_max - 0.01))
        sat_ratio = n_saturated / max(len(u_ctrl), 1) * 100
        log.info("  Saturation     : %d steps (%.1f%%), u_max=%.0f N, peak |u|=%.1f N",
                 n_saturated, sat_ratio, u_max, peak_u)
    else:
        log.info("  Peak |u|       : %.1f N (no saturation limit)", peak_u)
    log.info("  Disturbance    : [%+.2f, %+.2f] N", u_dist.min(), u_dist.max())

    # Frequency-domain data (if available)
    if freq_data is not None:
        pm = freq_data.get("phase_margin", np.nan)
        gm = freq_data.get("gain_margin_dB", np.nan)
        os_pct = freq_data.get("overshoot", np.nan)
        ts = freq_data.get("t_settle", np.nan)
        Ms = freq_data.get("Ms", np.nan)
        Mt = freq_data.get("Mt", np.nan)
        log.info("  Phase margin   : %.2f deg", pm)
        log.info("  Gain margin    : %.2f dB", gm)
        log.info("  Overshoot      : %.2f %%", os_pct)
        log.info("  Settling time  : %.4f s", ts)
        log.info("  Ms (peak |S|)  : %.3f", Ms)
        log.info("  Mt (peak |T|)  : %.3f", Mt)

    log.info("=" * 60)
