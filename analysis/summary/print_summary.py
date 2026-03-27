"""Print a concise simulation summary to the console."""

import numpy as np

from utils.logger import get_logger

log = get_logger()


def print_summary(q, dq, state, u_ctrl, u_dist, freq_data,
                  u_max=None, u_raw_peak=None, n_saturated=None) -> None:
    """Print key simulation metrics."""
    log.info("=" * 60)
    log.info("  SIMULATION SUMMARY")
    log.info("=" * 60)

    # State ranges (NaN-safe for diverged simulations)
    log.info("  Cart position : [%+.4f, %+.4f] m", np.nanmin(q[:, 0]), np.nanmax(q[:, 0]))
    log.info("  Cart velocity : [%+.4f, %+.4f] m/s", np.nanmin(dq[:, 0]), np.nanmax(dq[:, 0]))

    for i, key in enumerate(["dth1", "dth2", "dth3"], 1):
        arr = state[key]
        log.info("  dtheta%d       : [%+.4f, %+.4f] deg",
                 i, np.degrees(np.nanmin(arr)), np.degrees(np.nanmax(arr)))

    # Control effort (NaN-safe)
    valid_u = u_ctrl[np.isfinite(u_ctrl)]
    if len(valid_u) > 0:
        log.info("  Control force  : [%+.2f, %+.2f] N", valid_u.min(), valid_u.max())
    else:
        log.info("  Control force  : N/A (simulation diverged)")

    # Saturation statistics (from loop-level tracking)
    if u_raw_peak is not None and u_max is not None and u_max < 1e20:
        n_sat = n_saturated if n_saturated is not None else 0
        sat_ratio = n_sat / max(len(valid_u), 1) * 100
        log.info("  Saturation     : %d steps (%.1f%%), u_max=%.0f N, peak |u_raw|=%.1f N",
                 n_sat, sat_ratio, u_max, u_raw_peak)
    elif u_raw_peak is not None:
        log.info("  Peak |u_raw|   : %.1f N (no saturation limit)", u_raw_peak)

    log.info("  Disturbance    : [%+.2f, %+.2f] N", np.nanmin(u_dist), np.nanmax(u_dist))

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
