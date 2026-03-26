"""Shared axis styling helpers."""


def apply_grid(ax):
    """Add a light grid to *ax*."""
    ax.grid(True, alpha=0.3)


def apply_zero_line(ax):
    """Draw a dashed zero reference line on *ax*."""
    ax.axhline(0, color="k", ls="--", lw=0.5)
