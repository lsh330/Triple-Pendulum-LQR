"""All 8 equilibrium configurations of the triple pendulum on a cart.

Coordinate conventions
----------------------
- theta_i: relative joint angle of link i
- phi_i: absolute angle of link i (phi1=theta1, phi2=theta1+theta2, phi3=theta1+theta2+theta3)
- phi=0  -> link pointing downward (stable)
- phi=pi -> link pointing upward   (unstable)

PE sign convention
------------------
  PE_code = sum_i(m_i * g * h_i)   where h_i uses cos(phi_i) > 0 when downward
  V_physical = -PE_code             (standard gravitational PE)
  Physical energy: E = KE - PE_code = KE + V_physical
"""
import numpy as np


# Absolute angles (phi1, phi2, phi3) for each equilibrium configuration.
# phi=0 -> Down, phi=pi -> Up
EQUILIBRIUM_CONFIGS = {
    "DDD": (0.0,      0.0,      0.0),
    "DDU": (0.0,      0.0,      np.pi),
    "DUD": (0.0,      np.pi,    0.0),
    "DUU": (0.0,      np.pi,    np.pi),
    "UDD": (np.pi,    0.0,      0.0),
    "UDU": (np.pi,    0.0,      np.pi),
    "UUD": (np.pi,    np.pi,    0.0),
    "UUU": (np.pi,    np.pi,    np.pi),
}


def equilibrium(config_name="UUU"):
    """Return q_eq = [x, theta1, theta2, theta3] for the named configuration.

    Default "UUU" maintains backward compatibility with the original code.

    Parameters
    ----------
    config_name : str
        One of "DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU".

    Returns
    -------
    np.ndarray, shape (4,)
        Equilibrium configuration vector [x_eq, theta1_eq, theta2_eq, theta3_eq].
    """
    if config_name not in EQUILIBRIUM_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Choose from: {list(EQUILIBRIUM_CONFIGS.keys())}"
        )
    phi1, phi2, phi3 = EQUILIBRIUM_CONFIGS[config_name]
    # Convert absolute angles to relative joint angles
    theta1 = phi1
    theta2 = phi2 - phi1
    theta3 = phi3 - phi2
    return np.array([0.0, theta1, theta2, theta3])


def all_equilibria():
    """Return dict mapping config_name -> q_eq array for all 8 configurations."""
    return {name: equilibrium(name) for name in EQUILIBRIUM_CONFIGS}


def equilibrium_potential_energy(cfg, config_name):
    """Compute PHYSICAL potential energy V_physical = -PE_code at the given equilibrium.

    At equilibrium KE = 0, so V_physical equals the total physical energy.

    Parameters
    ----------
    cfg : SystemConfig
        System configuration providing m1, m2, m3, L1, L2, L3, g.
    config_name : str
        Equilibrium configuration name.

    Returns
    -------
    float
        Physical potential energy V_physical (J). Negative for downward configs,
        positive for upward configs.
    """
    phi1, phi2, phi3 = EQUILIBRIUM_CONFIGS[config_name]
    g = cfg.g
    # Heights of each link's center of mass (cart at height 0)
    h1 = (cfg.L1 / 2.0) * np.cos(phi1)
    h2 = cfg.L1 * np.cos(phi1) + (cfg.L2 / 2.0) * np.cos(phi2)
    h3 = (cfg.L1 * np.cos(phi1) + cfg.L2 * np.cos(phi2)
          + (cfg.L3 / 2.0) * np.cos(phi3))
    pe_code = cfg.m1 * g * h1 + cfg.m2 * g * h2 + cfg.m3 * g * h3
    return -pe_code  # Physical V = -PE_code


def all_potential_energies(cfg):
    """Return dict mapping config_name -> V_physical (J) for all configurations."""
    return {name: equilibrium_potential_energy(cfg, name)
            for name in EQUILIBRIUM_CONFIGS}


def config_index(config_name):
    """Return integer index (0-7) for the configuration name."""
    names = list(EQUILIBRIUM_CONFIGS.keys())
    return names.index(config_name)


def config_name_from_index(idx):
    """Return config name string from integer index (0-7)."""
    return list(EQUILIBRIUM_CONFIGS.keys())[idx]
