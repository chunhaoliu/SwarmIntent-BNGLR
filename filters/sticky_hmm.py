"""Intent-level multiple-model filter (IMM-style sticky HMM).

Each frame the intent persists with probability 1 - h and jumps uniformly to
another intent otherwise. This is CRIF with the excursion states removed
(h_e = 0), the first ablation.
"""
import numpy as np

from .common import scaled_likelihood


def posterior(e, h):
    B, T, K = e.shape
    p = np.full((B, K), 1.0 / K)
    out = np.empty((B, T, K))
    for t in range(T):
        p = p * (1.0 - h) + (1.0 - p) * h / (K - 1)
        p = p * scaled_likelihood(e[:, t])
        p /= p.sum(-1, keepdims=True)
        out[:, t] = p
    return out
