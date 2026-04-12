"""Pre-allocated data logger for simulation recording."""
import numpy as np


class DataLogger:
    """Records simulation data with pre-allocated numpy arrays."""

    __slots__ = ('_time', '_states', '_velocities', '_inputs',
                 '_capacity', '_count')

    def __init__(self, capacity=100000):
        self._capacity = capacity
        self._count = 0
        self._time = np.zeros(capacity)
        self._states = np.zeros((capacity, 4))
        self._velocities = np.zeros((capacity, 4))
        self._inputs = np.zeros(capacity)

    def on_step(self, t, q, dq, u):
        if self._count >= self._capacity:
            self._expand()
        self._time[self._count] = t
        self._states[self._count] = q
        self._velocities[self._count] = dq
        self._inputs[self._count] = u
        self._count += 1

    def _expand(self):
        new_cap = self._capacity * 2
        # 1-D arrays
        for attr in ('_time', '_inputs'):
            old = getattr(self, attr)
            new = np.zeros(new_cap)
            new[:self._capacity] = old
            setattr(self, attr, new)
        # 2-D arrays
        for attr in ('_states', '_velocities'):
            old = getattr(self, attr)
            new = np.zeros((new_cap, old.shape[1]))
            new[:self._capacity] = old
            setattr(self, attr, new)
        self._capacity = new_cap

    @property
    def time(self):
        return self._time[:self._count]

    @property
    def states(self):
        return self._states[:self._count]

    @property
    def velocities(self):
        return self._velocities[:self._count]

    @property
    def inputs(self):
        return self._inputs[:self._count]

    @property
    def count(self):
        return self._count
