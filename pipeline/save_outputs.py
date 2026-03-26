"""Save all figures and animation to images/ directory."""

import os

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def ensure_dir():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_figure(fig, name, dpi=150):
    """Save a matplotlib figure as PNG."""
    ensure_dir()
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")


def save_animation(ani, name, fps=30):
    """Save a matplotlib animation as GIF."""
    ensure_dir()
    path = os.path.join(IMAGES_DIR, f"{name}.gif")
    try:
        ani.save(path, writer="pillow", fps=fps)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  GIF save failed ({e}). Install pillow: pip install pillow")
