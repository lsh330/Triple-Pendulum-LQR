import numpy as np
from numba import njit


@njit(cache=True)
def compute_trig(q):
    th1 = q[1]
    th2 = q[2]
    th3 = q[3]
    c1 = np.cos(th1)
    c12 = np.cos(th1 + th2)
    c123 = np.cos(th1 + th2 + th3)
    c2 = np.cos(th2)
    c23 = np.cos(th2 + th3)
    c3 = np.cos(th3)
    s1 = np.sin(th1)
    s12 = np.sin(th1 + th2)
    s123 = np.sin(th1 + th2 + th3)
    return c1, c12, c123, c2, c23, c3, s1, s12, s123
