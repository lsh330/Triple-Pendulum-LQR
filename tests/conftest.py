"""Shared test fixtures for triple pendulum simulation."""
import matplotlib
matplotlib.use("Agg")  # CI/CD 및 headless 환경에서 Tcl/Tk 의존 제거

import pytest
import numpy as np
from parameters.config import SystemConfig


@pytest.fixture(scope="session")
def cfg():
    """Standard Medrano-Cerda configuration."""
    return SystemConfig(mc=3.0, m1=1.323, m2=1.389, m3=0.8655,
                        L1=0.402, L2=0.332, L3=0.720)


@pytest.fixture(scope="session")
def p(cfg):
    """Packed parameter vector."""
    return cfg.pack()


@pytest.fixture(scope="session")
def q_eq_uuu(cfg):
    """UUU equilibrium state."""
    return cfg.equilibrium


@pytest.fixture(scope="session")
def lqr_data(cfg):
    """LQR gains for UUU equilibrium."""
    from control.lqr import compute_lqr_gains
    K, A, B, P, Q, R = compute_lqr_gains(cfg)
    return {'K': K, 'A': A, 'B': B, 'P': P, 'Q': Q, 'R': R}
