# NOTE: This array-based forward dynamics is kept for compatibility with:
#   - control/linearization/ (JIT Jacobian computation)
#   - analysis/region_of_attraction.py (via _run_loop -> rk4_step)
#   - control/ilqr.py (_rk4_step)
# The scalar-based forward_dynamics_fast.py is used for main simulation loops.
import numpy as np
from numba import njit

from dynamics.mass_matrix.assembly import mass_matrix
from dynamics.coriolis.christoffel import coriolis_vector
from dynamics.gravity.gravity_vector import gravity_vector
from dynamics.forward_dynamics.tau_assembly import assemble_tau
from dynamics.forward_dynamics.solve_acceleration import solve_acceleration


@njit(cache=True)
def forward_dynamics(q, dq, u, p):
    M = mass_matrix(q, p)
    C = coriolis_vector(q, dq, p)
    G = gravity_vector(q, p)
    tau = assemble_tau(u)

    rhs = tau - C - G

    ddq = solve_acceleration(M, rhs)
    return ddq
