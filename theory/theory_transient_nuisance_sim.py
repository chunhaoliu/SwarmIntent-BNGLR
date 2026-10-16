"""Sanity check: detecting a persistent bad change f0->fB while ignoring a
transient confusing change f0->fC lasting L<=D frames, then back to f0.
Gaussian means, unit variance. Bounded-nuisance GLR vs CUSUM vs unbounded (D=inf).
Conjecture: minimal delay k* = max(log(1/a_s)/D_B0, k_a),
k_a = log(1/a)/D_BC if <= D else D + (log(1/a) - D*D_BC)/D_B0."""
import numpy as np, sys
rng = np.random.default_rng(0)
muB, muC, D = 1.0, float(sys.argv[1]) if len(sys.argv) > 1 else 0.8, 20
T, nu, R, m = 160, 40, 1000, 120
DB0, DBC = muB**2 / 2, (muB - muC)**2 / 2

def llr(x):  # per-frame log fB/f0 and log fC/f0
    return muB * x - muB**2 / 2, muC * x - muC**2 / 2

def stat(x, Dn):
    """For each t: S0(t)=max_k W_k(t) (CUSUM), and for bounded-nuisance test the
    max over k of min-normalized pairs is needed -> return W[k,t], Lam[k,t]."""
    b, c = llr(x)
    Pb = np.concatenate([np.zeros((len(x), 1)), np.cumsum(b, 1)], 1)
    Pc = np.concatenate([np.zeros((len(x), 1)), np.cumsum(c, 1)], 1)
    Rn, TT = x.shape
    W = np.full((Rn, TT, TT), -np.inf); Lam = np.full((Rn, TT, TT), -np.inf)
    for k in range(TT):
        hi = min(TT, k + m)
        W[:, k, k:hi] = Pb[:, k + 1:hi + 1] - Pb[:, k:k + 1]
        # best nuisance explanation starting at k with duration <= Dn: max_{j in [k, min(t+1,k+Dn)]} Pc[j]-Pc[k]
        seg = Pc[:, k:hi + 1] - Pc[:, k:k + 1]
        if np.isfinite(Dn):
            seg = seg.copy(); seg[:, int(Dn) + 1:] = -np.inf
        cm = np.maximum.accumulate(seg, 1)[:, 1:]
        Lam[:, k, k:hi] = W[:, k, k:hi] - cm
    return W, Lam

def stop(W, Lam, b0, bc):
    ok = ((W >= b0) & (Lam >= bc)).any(1)  # over k
    t = np.where(ok.any(1), ok.argmax(1), W.shape[2])
    return t

def gen(kind):
    x = rng.standard_normal((R, T))
    if kind == 'switch': x[:, nu:] += muB
    if kind == 'feint': x[:, nu:nu + D] += muC
    return x

data = {k: gen(k) for k in ('steady', 'feint', 'switch')}
grid0 = np.arange(1, 10.01, 0.5); gridc = np.concatenate([[-np.inf], np.arange(0, 10.01, 0.5)])
results = {}
for name, Dn, use_c in [('CUSUM', D, False), ('bounded-GLR', D, True), ('unbounded(Chen-type)', np.inf, True)]:
    stats = {k: stat(v, Dn) for k, v in data.items()}
    pts = []
    for b0 in grid0:
        for bc in (gridc if use_c else [-np.inf]):
            s = {k: stop(*stats[k], b0, bc) for k in stats}
            pfa_s = (s['steady'] < T).mean(); pfa_f = ((s['feint'] >= nu) & (s['feint'] < nu + D + 30)).mean()
            sw = s['switch']; early = (sw < nu).mean()
            delay = np.minimum(sw[sw >= nu] - nu, T - nu).mean()
            pts.append((b0, bc, pfa_s, pfa_f, early, delay))
    results[name] = np.array(pts)
print(f'muC={muC}  D_B0={DB0:.3f} D_BC={DBC:.3f} D={D}')
for a in (0.1, 0.05, 0.02, 0.01):
    la = np.log(1 / a)
    ka = la / DBC if la / DBC <= D else D + (la - D * DBC) / DB0
    line = f'alpha={a:<5} k*_feint={ka:6.1f} |'
    for name, P in results.items():
        ok = (P[:, 2] <= 0.05) & (P[:, 3] <= a) & (P[:, 4] <= 0.05)
        line += f' {name}: ' + (f'{P[ok, 5].min():5.1f}' if ok.any() else '  inf')
    print(line)
