"""Default simulation pipeline parameters."""

T_END = 15.0              # simulation duration [s]
DT = 0.001                # time step [s]
IMPULSE = 5.0             # initial impulse on cart [N*s]
DIST_AMPLITUDE = 15.0     # band-limited noise RMS [N]
DIST_BANDWIDTH = 3.0      # noise cutoff frequency [Hz]
SEED = 42                 # random seed for disturbance
U_MAX = 200.0             # actuator force saturation limit [N]
USE_ILQR = False          # enable iLQR trajectory optimization
ILQR_HORIZON = 500        # iLQR planning horizon steps
ILQR_ITERATIONS = 10      # iLQR iteration count
GAIN_SCHEDULER = "1d"     # "1d" (cubic Hermite on theta1) or "3d" (trilinear on theta1,2,3)
ADAPTIVE_Q = False        # use inertia-scaled Q matrix instead of fixed default
