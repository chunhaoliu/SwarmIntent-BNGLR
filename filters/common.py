"""Pieces shared by every decision layer.

All layers read the same scaled log-likelihoods e[b, t, i] (flights x frames x
intents), start from the same initial declaration at frame t0, and report the
declared intent per frame plus the alarm frames.
"""
import numpy as np

T0 = 10


def emissions(logits, temperature, log_prior, kappa=1.0):
    """e_t(i) = kappa * (log softmax(z_t / T)_i - log pi_i)."""
    z = logits / temperature
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return kappa * (logp - log_prior)


def initial_declaration(e, t0=T0):
    return e[:, :t0 + 1].sum(1).argmax(-1)


def scaled_likelihood(e_t):
    return np.exp(e_t - e_t.max(-1, keepdims=True))


def decide_posterior(P, c0, eta, t0=T0):
    """Alarm when the monitored posterior mass off the declared intent exceeds eta;
    the new declaration is the posterior argmax."""
    B, T, _ = P.shape
    idx = np.arange(B)
    c = c0.copy()
    C = np.full((B, T), -1, np.int64)
    alarm = np.zeros((B, T), bool)
    C[:, t0] = c
    for t in range(t0 + 1, T):
        Pt = P[:, t]
        new = Pt.argmax(-1)
        a = (1.0 - Pt[idx, c] > eta) & (new != c)
        c = np.where(a, new, c)
        C[:, t], alarm[:, t] = c, a
    return C, alarm
