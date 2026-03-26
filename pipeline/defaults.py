"""Default simulation pipeline parameters."""

T_END = 15.0              # simulation duration [s]
DT = 0.002                # time step [s]
IMPULSE = 5.0             # initial impulse on cart [N*s]
DIST_AMPLITUDE = 15.0     # band-limited noise RMS [N]
DIST_BANDWIDTH = 3.0      # noise cutoff frequency [Hz]
SEED = 42                 # random seed for disturbance
