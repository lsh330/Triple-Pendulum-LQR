"""Shared colour definitions for visualisation."""

LINK_COLORS = ["tab:red", "tab:green", "tab:blue"]


def link_labels(cfg) -> list[str]:
    """Return 3 human-readable labels like 'L1=0.402m, m1=1.323kg'."""
    return [
        f"L1={cfg.L1:.3f}m, m1={cfg.m1:.3f}kg",
        f"L2={cfg.L2:.3f}m, m2={cfg.m2:.3f}kg",
        f"L3={cfg.L3:.3f}m, m3={cfg.m3:.3f}kg",
    ]
