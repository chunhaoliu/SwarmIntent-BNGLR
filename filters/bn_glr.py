"""Bounded-nuisance GLR (BN-GLR) on learned steady/excursion evidence.

For the declared intent c and each candidate j != c, from a candidate onset k:
  W_k^j(t)   = sum_{i=k}^{t} (eS[i,j] - eS[i,c] - drift)          switch vs steady
  Lam_k^j(t) = W_k^j(t) - max over segments [a,b] in [k,t], b-a+1 <= D,
               of sum (eE[i,c] - eS[i,c])                          switch vs best bounded feint
An alarm requires W >= b0 and Lam >= b1 at the same onset k; the declaration
moves to the firing j with the largest Lam, and all onsets restart after it.
The empty segment is allowed, so Lam <= W. Theory (iid, exact likelihoods):
theory/第1步_定理与证明.md. Neural evidence is correlated and only approximately
calibrated, so no guarantee transfers; the rule is a theory-guided design.
"""
import numpy as np
from .common import T0


def run(e_s, e_exc, c0, b0, b1, d_max, drift=0.0, t0=T0):
    e_s, e_exc = np.asarray(e_s, float), np.asarray(e_exc, float)
    if e_s.shape != e_exc.shape or e_s.ndim != 3 or d_max < 1:
        raise ValueError('Invalid BN-GLR emissions or nuisance duration')
    B, T, K = e_s.shape
    idx = np.arange(B)
    c = np.asarray(c0).copy()
    start = np.full(B, t0 + 1)
    W = np.zeros((B, T, K))
    M = np.zeros((B, T))
    P = np.zeros((B, T + 1))
    C = np.full((B, T), -1, np.int64)
    alarm = np.zeros((B, T), bool)
    C[:, t0] = c
    ks = np.arange(T)
    for t in range(t0 + 1, T):
        es_c = e_s[idx, t, c]
        bt = e_s[:, t, :] - es_c[:, None] - drift
        P[:, t + 1] = P[:, t] + e_exc[idx, t, c] - es_c
        W[:, t, :] = 0.0
        M[:, t] = 0.0
        W[:, t0 + 1:t + 1, :] += bt[:, None, :]
        lo = max(t0 + 1, t + 1 - int(d_max))
        suffix = np.minimum.accumulate(P[:, lo:t + 1][:, ::-1], 1)[:, ::-1]
        k = ks[t0 + 1:t + 1]
        a0 = np.maximum(k, lo) - lo
        M[:, t0 + 1:t + 1] = np.maximum(M[:, t0 + 1:t + 1], P[:, t + 1:t + 2] - suffix[:, a0])
        Wk, Mk = W[:, t0 + 1:t + 1, :], M[:, t0 + 1:t + 1, None]
        lam = Wk - Mk
        ok = (Wk >= b0) & (lam >= b1) & (k[None, :, None] >= start[:, None, None])
        ok[idx, :, c] = False
        score = np.where(ok, lam, -np.inf).max(1)
        new = score.argmax(-1)
        fire = np.isfinite(score[idx, new])
        c = np.where(fire, new, c)
        start = np.where(fire, t + 1, start)
        C[:, t], alarm[:, t] = c, fire
    return C, alarm
