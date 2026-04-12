"""Organized output directory management."""
import numpy as np
from pathlib import Path
from datetime import datetime


class OutputManager:
    """Manages output directories for plots, animations, and data."""

    __slots__ = ('_base',)

    SUBDIRS = ('plots', 'animations', 'data', 'logs')

    def __init__(self, base_dir="output"):
        self._base = Path(base_dir)
        for subdir in self.SUBDIRS:
            (self._base / subdir).mkdir(parents=True, exist_ok=True)

    def _timestamped(self, name, ext):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{name}.{ext}"

    def plot_path(self, name, ext="png"):
        return self._base / "plots" / self._timestamped(name, ext)

    def animation_path(self, name, ext="gif"):
        return self._base / "animations" / self._timestamped(name, ext)

    def data_path(self, name, ext="npz"):
        return self._base / "data" / self._timestamped(name, ext)

    def log_path(self, name, ext="log"):
        return self._base / "logs" / self._timestamped(name, ext)

    def save_trajectory(self, result, name="trajectory"):
        """Save simulation trajectory as compressed numpy archive."""
        path = self.data_path(name, "npz")
        save_dict = {}
        for key in ('t', 'q', 'dq', 'u', 'mode', 'stage', 'energy'):
            if key in result and result[key] is not None:
                save_dict[key] = result[key]
        np.savez_compressed(str(path), **save_dict)
        return path
