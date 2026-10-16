"""Idealized-model checks of the bounded-nuisance GLR (BN-GLR) theory.

Model: f0=N(0,1), fB=N(1,1), fC=N(muC,1). A feint is fC for L<=D frames then f0.
I0=D(fB||f0)=1/2, I1=D(fB||fC)=(1-muC)^2/2. Theory: minimal delay
k* = max(log(gamma)/I0, k_alpha) with mu(k_alpha)=log(1/alpha) (see theory notes).

Experiments (no project data is used):
  frontier  - best delay per (alpha) for BN-GLR, CUSUM and an unbounded (D=inf) GLR
  scaling   - theory thresholds b0=log(gamma), b1=log(2N/alpha); delay/k* as the problem scales
  robust    - random feint onset, L<D, and feint-then-switch at the theory thresholds
Usage: python scripts/theory_bn_glr_sim.py {frontier|scaling|robust} [--out file.json]
"""
import argparse
import json
import numpy as np

MU_B = 1.0


def llr(x, mu_c):
    return MU_B * x - MU_B ** 2 / 2, mu_c * x - mu_c ** 2 / 2


def stats(x, mu_c, D):
    """W[r,k,t] and Lam[r,k,t] for all k<=t (float32). Lam uses the composite nuisance null:
    best segment of length <=D anywhere inside [k,t] (empty segment allowed)."""
    b, c = llr(x, mu_c)
    R, T = x.shape
    Pb = np.concatenate([np.zeros((R, 1)), np.cumsum(b, 1)], 1)
    Pc = np.concatenate([np.zeros((R, 1)), np.cumsum(c, 1)], 1)
    W = np.full((R, T, T), -np.inf, np.float32)
    Lam = np.full((R, T, T), -np.inf, np.float32)
    M = np.zeros((R, T))  # M[:,k] running max segment sum of c within [k,t]
    Dn = T + 1 if not np.isfinite(D) else int(D)
    for t in range(T):
        # min of Pc[a] for a in [max(k, t+1-Dn), t+1) using prefix index a (segment a..t)
        lo = max(0, t + 1 - Dn)
        win = Pc[:, lo:t + 2]                       # Pc indices lo..t+1
        # suffix minima so that for each start-limit a0 we get min_{a>=a0} over window (excluding last)
        suf = np.minimum.accumulate(win[:, :-1][:, ::-1], 1)[:, ::-1]   # for a in lo..t
        ks = np.arange(t + 1)
        a0 = np.maximum(ks, lo) - lo                # index into suf
        best_new = Pc[:, t + 1:t + 2] - suf[:, a0]
        M[:, :t + 1] = np.maximum(M[:, :t + 1], best_new)
        Wt = Pb[:, t + 1:t + 2] - Pb[:, :t + 1]
        W[:, :t + 1, t] = Wt
        Lam[:, :t + 1, t] = Wt - M[:, :t + 1]
    return W, Lam


def first_alarm(W, Lam, b0, b1):
    ok = ((W >= b0) & (Lam >= b1)).any(1)
    T = W.shape[2]
    return np.where(ok.any(1), ok.argmax(1), T)


def k_alpha(la, D, I0, I1):
    if I1 > 0 and la / I1 <= D:
        return la / I1
    return D + (la - D * I1) / I0


def gen(rng, R, T, kind, mu_c, nu, D, L=None, onset_jitter=0, gap=0):
    x = rng.standard_normal((R, T))
    L = D if L is None else L
    on = nu + (rng.integers(-onset_jitter, onset_jitter + 1, R) if onset_jitter else np.zeros(R, int))
    if kind == 'switch':
        x[:, nu:] += MU_B
    if kind in ('feint', 'feint_switch'):
        for r in range(R):
            x[r, on[r]:on[r] + L] += mu_c
    if kind == 'feint_switch':
        x[:, nu + L + gap:] += MU_B
    return x


def frontier(args):
    rng = np.random.default_rng(1)
    T, nu, D, R = 160, 40, 20, args.runs
    out = []
    for mu_c in (0.8, 0.65, 0.5):
        I0, I1 = 0.5, (1 - mu_c) ** 2 / 2
        data = {k: gen(rng, R, T, k, mu_c, nu, D) for k in ('steady', 'feint', 'switch')}
        curves = {}
        for name, Dn, use_c in [('CUSUM', D, False), ('BN-GLR', D, True), ('GLR-unbounded', np.inf, True)]:
            st = {k: stats(v, mu_c, Dn) for k, v in data.items()}
            pts = []
            for b0 in np.arange(1, 10.01, 0.5):
                for b1 in ([-np.inf] if not use_c else np.concatenate([[-np.inf], np.arange(0, 10.01, 0.5)])):
                    s = {k: first_alarm(*st[k], b0, b1) for k in st}
                    pts.append(dict(b0=float(b0), b1=float(b1), steady=float((s['steady'] < T).mean()),
                                    feint=float(((s['feint'] >= nu) & (s['feint'] < nu + D + 30)).mean()),
                                    early=float((s['switch'] < nu).mean()),
                                    delay=float(np.minimum(s['switch'][s['switch'] >= nu] - nu, T - nu).mean())))
            curves[name] = pts
            del st
        for a in (0.1, 0.05, 0.02, 0.01):
            row = dict(mu_c=mu_c, I1=I1, alpha=a, k_alpha=k_alpha(np.log(1 / a), D, I0, I1))
            for name, pts in curves.items():
                ok = [p['delay'] for p in pts if p['steady'] <= .05 and p['early'] <= .05 and p['feint'] <= a]
                row[name] = min(ok) if ok else None
            out.append(row)
            print(row, flush=True)
    return out


def scaling(args):
    rng = np.random.default_rng(2)
    mu_c = 0.684  # I1 ~ 0.05
    I0, I1 = 0.5, (1 - mu_c) ** 2 / 2
    out = []
    for s in (1, 2, 4, 8):
        D, la, lg = 5 * s, 1.5 * s, 2.0 * s          # D frames, log(1/alpha), log(gamma)
        alpha, gamma = np.exp(-la), np.exp(lg)
        ks = max(lg / I0, k_alpha(la, D, I0, I1))
        nu = 10
        T = int(nu + 2.5 * ks + 30)
        N = T + D + 30
        b0, b1 = lg, np.log(2 * N / alpha)
        R = args.runs
        res = dict(scale=s, D=D, alpha=float(alpha), log_gamma=lg, k_star=ks, b0=b0, b1=float(b1), T=T)
        for kind in ('feint', 'switch'):
            x = gen(rng, R, T, kind, mu_c, nu, D)
            W, Lam = stats(x, mu_c, D)
            st = first_alarm(W, Lam, b0, b1)
            if kind == 'feint':
                res['feint_fa'] = float(((st >= nu) & (st < nu + D + 30)).mean())
                res['pre_onset_alarm'] = float((st < nu).mean())
            else:
                d = np.minimum(st[st >= nu] - nu, T - nu)
                res['delay'] = float(d.mean())
                res['delay_over_kstar'] = float(d.mean() / ks)
                res['k_star_thresholds'] = max(b0 / I0, k_alpha(b1, D, I0, I1))
                res['delay_over_kstar_thresholds'] = float(d.mean() / res['k_star_thresholds'])
                res['missed'] = float((st >= T).mean())
            del W, Lam
        out.append(res)
        print(res, flush=True)
    return out


def robust(args):
    rng = np.random.default_rng(3)
    mu_c, D, nu, T, R = 0.65, 20, 40, 160, args.runs
    I0, I1 = 0.5, (1 - mu_c) ** 2 / 2
    alpha = 0.05
    N = T + D + 30
    b0, b1 = np.log(200.), np.log(2 * N / alpha)
    out = []
    cases = [('feint L=D', dict(kind='feint')), ('feint L=D/2', dict(kind='feint', L=D // 2)),
             ('feint random onset', dict(kind='feint', onset_jitter=10)),
             ('switch', dict(kind='switch')), ('feint then switch (gap 5)', dict(kind='feint_switch', gap=5))]
    for name, kw in cases:
        x = gen(rng, R, T, mu_c=mu_c, nu=nu, D=D, **kw)
        W, Lam = stats(x, mu_c, D)
        st = first_alarm(W, Lam, b0, b1)
        row = dict(case=name, alarm_before_window=float((st < nu - 10).mean()))
        if kw['kind'] == 'feint':
            row['feint_fa'] = float(((st >= nu - 10) & (st < nu + D + 30)).mean())
        else:
            onset = nu if kw['kind'] == 'switch' else nu + kw.get('L', D) + kw.get('gap', 0)
            ok = st >= onset
            row['delay_from_switch'] = float(np.minimum(st[ok] - onset, T - onset).mean())
            row['alarm_before_switch'] = float((st < onset).mean())
        row['theory_k_star'] = max(b0 / I0, k_alpha(b1, D, I0, I1))
        out.append(row)
        print(row, flush=True)
        del W, Lam
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', choices=['frontier', 'scaling', 'robust'])
    ap.add_argument('--runs', type=int, default=1000)
    ap.add_argument('--out')
    a = ap.parse_args()
    res = dict(frontier=frontier, scaling=scaling, robust=robust)[a.experiment](a)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)
