"""Energy-based swing-up controller for the triple pendulum.

Control law
-----------
  u = k_energy * (E_target - E_current) * dx_cart

Lyapunov stability argument:
  V_lyap = 0.5 * (E - E*)^2  >= 0
  dV_lyap/dt = (E - E*) * dE/dt
             = (E - E*) * u * dx_cart       (dE/dt = F_cart * v_cart)
             = (E - E*) * k_e*(E*-E)*dx^2
             = -k_e * (E - E*)^2 * dx^2  <= 0

so V_lyap is non-increasing. The control drives E -> E*.
"""
import numpy as np
from numba import njit
from control.swing_up.energy_computation import total_energy_scalar


@njit(cache=True)
def energy_swing_up_control(sq0, sq1, sq2, sq3, sdq0, sdq1, sdq2, sdq3,
                             p, E_target, k_energy, u_max):
    """Compute energy-based swing-up control input.

    Parameters
    ----------
    sq0..sq3 : float
        Configuration scalars [x, theta1, theta2, theta3].
    sdq0..sdq3 : float
        Velocity scalars [dx, dtheta1, dtheta2, dtheta3].
    p : 1-D array (13,)
        Packed parameter vector.
    E_target : float
        Target physical energy E* at the desired equilibrium (J).
    k_energy : float
        Energy shaping gain (> 0).
    u_max : float
        Saturation limit for the control force (N).

    Returns
    -------
    float
        Saturated control force u (N).
    """
    E_current = total_energy_scalar(sq0, sq1, sq2, sq3,
                                    sdq0, sdq1, sdq2, sdq3, p)

    dx = sdq0  # Cart velocity (used as dissipation direction)
    u = k_energy * (E_target - E_current) * dx

    # Saturation
    if u > u_max:
        u = u_max
    elif u < -u_max:
        u = -u_max

    return u
