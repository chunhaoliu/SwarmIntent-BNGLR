"""Frame-wise argmax with M-of-N confirmation (common engineering practice)."""
import numpy as np

from .common import T0


def run(e, c0, m, n, t0=T0, mode="off_current", gate=None):
    """Legacy off-current confirmation or candidate-wise voting with no tie alarm.

    gate[b,t] optionally permits an alarm; it never changes the vote window.
    """
    B, T, K = e.shape
    if not 1 <= m <= n or mode not in ("off_current", "candidate"):
        raise ValueError("Invalid M/N or confirmation mode")
    if gate is not None and gate.shape != (B, T):
        raise ValueError("Invalid alarm gate shape")
    idx = np.arange(B)
    onehot = np.eye(K, dtype=np.int64)[e.argmax(-1)]
    cum = np.concatenate([np.zeros((B, 1, K), np.int64), onehot.cumsum(1)], 1)
    c = c0.copy()
    C = np.full((B, T), -1, np.int64)
    alarm = np.zeros((B, T), bool)
    C[:, t0] = c
    for t in range(t0 + 1, T):
        counts = cum[:, t + 1] - cum[:, max(0, t + 1 - n)]
        a = counts.sum(-1) - counts[idx, c] >= m
        counts[idx, c] = -1
        if mode == "candidate":
            maximum = counts.max(-1)
            a = (maximum >= m) & ((counts == maximum[:, None]).sum(-1) == 1)
        if gate is not None:
            a &= gate[:, t]
        c = np.where(a, counts.argmax(-1), c)
        C[:, t], alarm[:, t] = c, a
    return C, alarm
