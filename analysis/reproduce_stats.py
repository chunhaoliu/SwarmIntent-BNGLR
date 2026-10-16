"""Recompute the paper's test-set numbers from the per-flight outcome files and check them against the
stored statistics.

For every study the per-flight files (results/*/test/*_flights.npz) hold one boolean or float per flight
and method: any_fa (any false alarm), detected (switch detected), delay (censored delay, s), false_iso.
The per-seed JSON files hold the flight kinds and the summaries; the stats.json files hold the stored
paired comparisons. This script
  1. prints method means over seeds (feint and steady false-alarm rate, delay, detection, feint->switch
     detection) for each study, radar and delay budget (Tables 3-7 of the paper);
  2. recomputes every paired comparison on feint flights (discordant counts n10/n01 and the two-sided
     exact McNemar p-value, pooled over seeds) and checks it against the stored value.
Bootstrap intervals are not re-derived here (they depend on the resampling order); see stats.json.

  python analysis/reproduce_stats.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

R = Path(__file__).resolve().parents[1] / 'results'
FT, E2, SH, TK = R / 'final_test_v1', R / 'final_e2e_v1', R / 'final_shift_v1', R / 'final_tf2k_v1'
SEEDS = [42, 123, 456, 789, 1024]
checked, failed = 0, 0


def jl(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0)


def check(name, got, stored):
    global checked, failed
    checked += 1
    ok = got[0] == stored[0] and got[1] == stored[1] and abs(got[2] - stored[2]) < 1e-9
    if not ok:
        failed += 1
    print(f'  {"ok " if ok else "BAD"} {name}: n10/n01 {got[0]}/{got[1]} p={got[2]:.3g}' + ('' if ok else f'  (stored {stored})'))


def summaries(rows):
    keys = (('feint', lambda s: s['feint']['pfa']), ('steady', lambda s: s['steady']['pfa']),
            ('delay', lambda s: s['switch']['delay_censored_mean']), ('det', lambda s: s['switch']['det']),
            ('F->S', lambda s: s['feint_switch']['det']))
    ok = [r for r in rows if r]
    return len(ok), {k: float(np.mean([f(r) for r in ok])) for k, f in keys} if ok else {}


def print_means(title, per_seed):
    """per_seed: {seed: {method@L: summary or None}}"""
    print(f'\n== {title}')
    names = sorted({m for v in per_seed.values() for m in v})
    for m in names:
        n, v = summaries([per_seed[s].get(m) for s in per_seed])
        if n:
            print(f'  {m:32s} seeds {n}/5  ' + '  '.join(f'{k} {x:.3f}' if k != 'delay' else f'{k} {x:.1f}' for k, x in v.items()))


# ---------------------------------------------------------------- final test (main, kclass, mixed)
st = jl(FT / 'stats.json')['comparisons']
for study, radars, seeds in (('main', {'degraded': (24, 26), 'nominal': (22, 24)}, SEEDS),
                             ('kclass', {'degraded': (24, 26), 'nominal': (22, 24)}, SEEDS),
                             ('mixed', {'degraded': (24, 26)}, [42, 123, 456])):
    for radar, Ls in radars.items():
        js = {s: jl(FT / f'test/{study}_seed{s}.json')['radars'][radar] for s in seeds}
        print_means(f'final test, {study} study, {radar} radar',
                    {s: {k: (v['summary'] if v else None) for k, v in j['methods'].items()} for s, j in js.items()})
        for key, v in st.items():
            st_, r_, L, pair = key.split('|')
            if st_ != study or r_ != radar: continue
            a, b = pair.split(' vs ')
            A, B = [], []
            for s in seeds:
                if js[s]['methods'].get(f'{a}@{L}') is None or js[s]['methods'].get(f'{b}@{L}') is None: continue
                F = np.load(FT / f'test/{study}_seed{s}_{radar}_flights.npz'); fe = np.array(js[s]['kinds']) == 'feint'
                A.append(F[f'{a}@{L}|any_fa'][fe]); B.append(F[f'{b}@{L}|any_fa'][fe])
            check(f'{study} {radar} {L}s {pair}', mcnemar(np.concatenate(A), np.concatenate(B)), (v['n10'], v['n01'], v['mcnemar_p']))

# ---------------------------------------------------------------- end-to-end classifiers (vs BN-GLR of the final test)
st = jl(E2 / 'stats.json')
for radar, Ls in (('degraded', (24, 26)), ('nominal', (22, 24))):
    js = {s: jl(E2 / f'test/seed{s}.json')['radars'][radar] for s in SEEDS}
    print_means(f'end-to-end classifiers, {radar} radar', {s: {k: (v['summary'] if v else None) for k, v in j['methods'].items()} for s, j in js.items()})
    for L in Ls:
        for key, v in st.items():
            r_, L_, m = key.split('|')
            if r_ != radar or float(L_) != L or not v['vs_bnglr']['seeds']: continue
            A, B = [], []
            for s in SEEDS:
                if js[s]['methods'].get(f'{m}@{L}') is None: continue
                E = np.load(E2 / f'test/seed{s}_{radar}_flights.npz'); F = np.load(FT / f'test/main_seed{s}_{radar}_flights.npz')
                if f'BN-GLR@{L}|any_fa' not in F.files: continue
                fe = np.array(js[s]['kinds']) == 'feint'
                A.append(E[f'{m}@{L}|any_fa'][fe]); B.append(F[f'BN-GLR@{L}|any_fa'][fe])
            vb = v['vs_bnglr']
            check(f'e2e {radar} {L}s {m} vs BN-GLR', mcnemar(np.concatenate(A), np.concatenate(B)), (vb['n10'], vb['n01'], vb['mcnemar_p']))

# ---------------------------------------------------------------- distribution shift
st = jl(SH / 'stats.json')
print('\n== distribution shift (feint false-alarm rate, mean over seeds)')
for key, row in st.items():
    cond, L = key.split('@')
    print(f'  {key:14s} ' + '  '.join(f'{m.split("-")[0]} {v["feint"]:.3f}' for m, v in row.items() if 'vs' not in m and v['feint'] is not None))
    for base in ('Sticky-marginal', 'CUSUM-marginal'):
        A, B = [], []
        for s in SEEDS:
            Z = np.load(SH / f'seed{s}_flights.npz'); ka, kb = f'{cond}|BN-GLR@{L}|any_fa', f'{cond}|{base}@{L}|any_fa'
            if ka not in Z.files or kb not in Z.files: continue
            fe = np.array(jl(SH / f'seed{s}.json')['conds'][cond]['kinds']) == 'feint'
            A.append(Z[ka][fe]); B.append(Z[kb][fe])
        v = row[f'BN-GLR vs {base}']
        check(f'shift {key} BN-GLR vs {base}', mcnemar(np.concatenate(A), np.concatenate(B)), (v['n10'], v['n01'], v['p']))

# ---------------------------------------------------------------- encoder x rule on fresh flights
st = jl(TK / 'stats.json')['contrasts']
for radar in ('degraded', 'nominal'):
    js = {s: jl(TK / f'test/seed{s}.json')['radars'][radar] for s in SEEDS}
    print_means(f'encoder x rule, fresh flights, {radar} radar', {s: {k: (v['summary'] if v else None) for k, v in j['methods'].items()} for s, j in js.items()})
    for key, v in st.items():
        r_, L, tag = key.split('|')
        if r_ != radar: continue
        A, B = [], []
        for s in SEEDS:
            Z = np.load(TK / f'test/seed{s}_{radar}_flights.npz'); ka, kb = f'{v["a"]}@{L}|any_fa', f'{v["b"]}@{L}|any_fa'
            if ka not in Z.files or kb not in Z.files: continue
            fe = np.array(js[s]['kinds']) == 'feint'
            A.append(Z[ka][fe]); B.append(Z[kb][fe])
        check(f'fresh {radar} {L}s {tag} {v["a"]} vs {v["b"]}', mcnemar(np.concatenate(A), np.concatenate(B)), (v['n10'], v['n01'], v['p']))

# ---------------------------------------------------------------- idealized model (Table 2)
print('\n== idealized Gaussian model (results/theory_bn_glr_v1/frontier.json)')
for row in jl(R / 'theory_bn_glr_v1/frontier.json'):
    if row['alpha'] in (0.05, 0.01):
        print(f'  I1 {row["I1"]:.3f} alpha {row["alpha"]:g} k_alpha {row["k_alpha"]:.1f}  BN-GLR {row["BN-GLR"]:.1f}  '
              f'persistent {row["GLR-unbounded"]:.1f}  CUSUM ' + ('--' if row['CUSUM'] is None else f'{row["CUSUM"]:.1f}'))

print(f'\n{checked} paired comparisons recomputed, {failed} mismatches')
raise SystemExit(1 if failed else 0)
