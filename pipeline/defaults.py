"""Default simulation pipeline parameters."""

T_END = 15.0              # simulation duration [s]
DT = 0.001                # time step [s]
IMPULSE = 5.0             # initial impulse on cart [N*s]
DIST_AMPLITUDE = 15.0     # band-limited noise RMS [N]
DIST_BANDWIDTH = 3.0      # noise cutoff frequency [Hz]
SEED = 42                 # random seed for disturbance
U_MAX = 200.0             # actuator force saturation limit [N]
INTEGRATOR = "rk4"           # "rk4" (fixed step) or "rk45" (adaptive)
RK45_ATOL = 1e-8             # absolute tolerance for RK45
RK45_RTOL = 1e-6             # relative tolerance for RK45
USE_ILQR = False              # enable iLQR trajectory optimization
ILQR_HORIZON = 500            # iLQR planning horizon steps
ILQR_ITERATIONS = 10          # iLQR iteration count
