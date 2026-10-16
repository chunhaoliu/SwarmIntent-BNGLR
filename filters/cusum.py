"""Multi-alternative CUSUM with drift, restarted after every alarm."""
import numpy as np

from .common import T0


def run(e, c0, drift, tau, t0=T0):
    B, T, K = e.shape
    idx = np.arange(B)
    c = c0.copy()
    S = np.zeros((B, K))
    C = np.full((B, T), -1, np.int64)
    alarm = np.zeros((B, T), bool)
    C[:, t0] = c
    for t in range(t0 + 1, T):
        S = np.maximum(0.0, S + e[:, t] - e[idx, t, c][:, None] - drift)
        S[idx, c] = 0.0
        a = S.max(-1) > tau
        c = np.where(a, S.argmax(-1), c)
        S[a] = 0.0
        C[:, t], alarm[:, t] = c, a
    return C, alarm
