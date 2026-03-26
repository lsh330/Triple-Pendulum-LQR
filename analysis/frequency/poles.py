"""Compute open-loop and closed-loop poles."""

import numpy as np


def compute_poles(A, A_cl) -> dict:
    """Return dict with poles_ol and poles_cl (eigenvalues)."""
    poles_ol = np.linalg.eigvals(A)
    poles_cl = np.linalg.eigvals(A_cl)
    return dict(poles_ol=poles_ol, poles_cl=poles_cl)
