"""Save all figures and animation to images/ directory."""

import os

from utils.logger import get_logger

log = get_logger()

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def ensure_dir():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_figure(fig, name, dpi=150):
    """Save a matplotlib figure as PNG."""
    ensure_dir()
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("  Saved: %s", path)


def save_animation(ani, name, fps=30):
    """Save a matplotlib animation as GIF."""
    ensure_dir()
    path = os.path.join(IMAGES_DIR, f"{name}.gif")
    try:
        ani.save(path, writer="pillow", fps=fps)
        log.info("  Saved: %s", path)
    except Exception as e:
        log.warning("  GIF save failed (%s). Install pillow: pip install pillow", e)
