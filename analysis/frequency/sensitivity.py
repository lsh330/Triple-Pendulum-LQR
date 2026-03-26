"""Compute sensitivity S(jw) and complementary sensitivity T(jw)."""

import numpy as np


def compute_sensitivity(L_jw) -> dict:
    """Return dict with S_jw, T_jw, Ms, Mt.

    S = 1 / (1 + L),  T = L / (1 + L).
    """
    S_jw = 1.0 / (1.0 + L_jw)
    T_jw = L_jw / (1.0 + L_jw)

    Ms = np.max(np.abs(S_jw))
    Mt = np.max(np.abs(T_jw))

    return dict(S_jw=S_jw, T_jw=T_jw, Ms=Ms, Mt=Mt)
