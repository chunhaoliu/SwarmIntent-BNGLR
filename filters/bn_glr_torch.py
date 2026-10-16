"""GPU BN-GLR: the recursion of filters/bn_glr.py batched over operating points and flights.

All points share D and drift; (b0, b1) vary per point. Evidence is float64 on the device so the
result matches filters/bn_glr.run (tests/test_bn_glr_torch.py checks equality on random inputs).
"""
import numpy as np
import torch
from .common import T0


def run_batch(e_s, e_exc, c0, b0s, b1s, d_max, drift=0.0, t0=T0, device='cuda'):
    """Returns C (P,B,T) int64 and alarm (P,B,T) bool numpy arrays, one slice per (b0[p], b1[p])."""
    f = dict(dtype=torch.float64, device=device)
    es = torch.as_tensor(np.asarray(e_s, float), **f)
    ee = torch.as_tensor(np.asarray(e_exc, float), **f)
    B, T, K = es.shape
    b0 = torch.as_tensor(np.asarray(b0s, float), **f)[:, None, None, None]
    b1 = torch.as_tensor(np.asarray(b1s, float), **f)[:, None, None, None]
    P = b0.shape[0]
    bi = torch.arange(B, device=device)[None].expand(P, B)
    c = torch.as_tensor(np.asarray(c0), device=device).long()[None].expand(P, B).clone()
    start = torch.full((P, B), t0 + 1, device=device, dtype=torch.long)
    W = torch.zeros((P, B, T, K), **f)
    M = torch.zeros((P, B, T), **f)
    Pc = torch.zeros((P, B, T + 1), **f)
    C = torch.full((P, B, T), -1, device=device, dtype=torch.long)
    alarm = torch.zeros((P, B, T), device=device, dtype=torch.bool)
    C[:, :, t0] = c
    ks = torch.arange(T, device=device)
    for t in range(t0 + 1, T):
        es_t = es[:, t, :][None].expand(P, B, K)
        es_c = torch.gather(es_t, 2, c[..., None])[..., 0]
        bt = es_t - es_c[..., None] - drift
        ee_c = torch.gather(ee[:, t, :][None].expand(P, B, K), 2, c[..., None])[..., 0]
        Pc[:, :, t + 1] = Pc[:, :, t] + ee_c - es_c
        W[:, :, t, :] = 0.0
        M[:, :, t] = 0.0
        W[:, :, t0 + 1:t + 1, :] += bt[:, :, None, :]
        lo = max(t0 + 1, t + 1 - int(d_max))
        seg = Pc[:, :, lo:t + 1]
        suffix = torch.flip(torch.cummin(torch.flip(seg, [2]), 2).values, [2])
        k = ks[t0 + 1:t + 1]
        a0 = torch.clamp(k, min=lo) - lo
        M[:, :, t0 + 1:t + 1] = torch.maximum(M[:, :, t0 + 1:t + 1], Pc[:, :, t + 1:t + 2] - suffix[:, :, a0])
        Wk = W[:, :, t0 + 1:t + 1, :]
        lam = Wk - M[:, :, t0 + 1:t + 1, None]
        ok = (Wk >= b0) & (lam >= b1) & (k[None, None, :, None] >= start[..., None, None])
        ok.scatter_(3, c[:, :, None, None].expand(P, B, ok.shape[2], 1), False)
        score = torch.where(ok, lam, torch.tensor(-np.inf, **f)).amax(2)
        new = score.argmax(-1)
        fire = torch.isfinite(torch.gather(score, 2, new[..., None])[..., 0])
        c = torch.where(fire, new, c)
        start = torch.where(fire, torch.full_like(start, t + 1), start)
        C[:, :, t], alarm[:, :, t] = c, fire
    return C.cpu().numpy(), alarm.cpu().numpy()
