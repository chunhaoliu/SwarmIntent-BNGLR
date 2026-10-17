# BN-GLR: swarm intent-switch detection under bounded-duration feints (core release)

Code and evaluation outputs for

> C. Liu et al., *Detecting UAV Swarm Intent Switches Under Bounded-Duration Feints: A Delay Bound and Learned
> Regime Evidence*, submitted to IEEE Transactions on Aerospace and Electronic Systems.

This core release contains the bounded-nuisance GLR (BN-GLR) test, the baseline decision layers, the
idealized-model simulations of Section VI-A, and the per-flight outcomes of every test-set
comparison in the paper, together with a script that recomputes the paired statistics. The radar-level
swarm, feint and tracker simulator, the trained encoders and the training scripts are not included; they
are available from the corresponding author on reasonable request.

## Contents

| Path | What |
|---|---|
| `filters/bn_glr.py` | BN-GLR, NumPy (prefix sums, recursive best feint segment; Eqs. (5)-(6)) |
| `filters/bn_glr_torch.py`, `exp/bn_glr_detect_torch.py` | GPU-batched BN-GLR over a threshold grid, identical outputs |
| `exp/bn_glr_detect.py` | detector wrapper and the operating-point grid used in the paper |
| `filters/cusum.py`, `sticky_hmm.py`, `m_of_n.py`, `window_glr.py` | baseline decision layers (CUSUM, sticky HMM, M-of-N, window GLR) |
| `tests/` | BN-GLR against brute-force enumeration and against CUSUM (b1 = -inf); GPU vs NumPy |
| `theory/theory_bn_glr_sim.py` | idealized Gaussian model: frontier (Table I), scaling, robustness |
| `theory/theory_transient_nuisance_sim.py` | delay lower bound vs simulated detectors |
| `results/final_test_v1/` | pre-registered final test (plan `results/final_test_plan_v1.md`): main, K-class ablation and whole-swarm studies |
| `results/final_e2e_v1/` | end-to-end classifiers (GraphGRU, causal Transformer), points frozen before scoring |
| `results/final_shift_v1/` | nine distribution-shift conditions at the frozen final-test points |
| `results/final_tf2k_v1/` | encoder x decision rule on fresh test flights (plan `results/final_tf2k_plan_v1.md`) |
| `results/theory_bn_glr_v1/` | stored outputs of the idealized-model simulations |
| `analysis/reproduce_stats.py` | recomputes method means and all 108 paired McNemar tests from the per-flight files |
| `GRIDS.md` | parameter grids of every decision layer and the end-to-end training settings |

Each `results/*/test/*.json` file holds, per seed and radar, the flight kinds (`steady`, `switch`,
`feint`, `feint_switch`), the frozen operating point of every method and its summary. The matching
`*_flights.npz` files hold one entry per flight and method: `any_fa` (any false alarm), `detected`,
`delay` (censored, s) and `false_iso`. `selected_points.json` and its SHA-256 record the operating
points frozen before the test flights were scored.

## Usage

Python 3.12; `pip install -r requirements.txt` (PyTorch is needed only for the GPU detector and its test).

```bash
python -m unittest discover -s tests -t .
python analysis/reproduce_stats.py
python theory/theory_bn_glr_sim.py frontier --runs 2000 --out frontier.json
python theory/theory_bn_glr_sim.py scaling --out scaling.json
```

`reproduce_stats.py` exits with status 0 when every recomputed comparison equals the stored one.

### Using BN-GLR on your own evidence

```python
from filters import bn_glr
# e_s, e_exc: (flights, scans, K) per-scan log evidence for steady and excursion regimes of each intent;
# c0: (flights,) initial declarations; t0: first monitored scan.
C, alarm = bn_glr.run(e_s, e_exc, c0, b0=20.0, b1=3.0, d_max=36, drift=1.0, t0=10)
```

`C[i, t]` is the declared intent and `alarm[i, t]` marks a declaration change. With `b1=-1e9` the test is
CUSUM on the steady evidence.

## License

MIT, see `LICENSE`.
