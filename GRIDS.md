# Parameter grids of the decision layers

Every method is tuned over its own grid by the same rule: an operating point is feasible if the one-sided
exact 95% upper bound of its false-alarm rate, over the steady and switch flights of each radar setting of
the threshold-selection split, is at most 0.05; at each delay budget the feasible point with the lowest
development feint false-alarm rate is used. All grids were fixed before the final test. The chosen points
are in `results/*/selected_points.json`.

| Method | Evidence | Grid |
|---|---|---|
| CUSUM | marginal `e_t` | drift δ ∈ {0, 0.5, 1, 2, 3}; threshold τ: 36 values log-spaced in [0.5, 400] |
| Sticky HMM | marginal `e_t` (ablation: steady `e^S`) | switch hazard h_s ∈ {1/30, 1/90, 1/270, 1/810, 1/2430}; posterior threshold η ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 0.99999, 0.999999}; evidence scale κ ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} |
| CRIF (P_S) | `e^S`, `e^E` | maximum excursion D_max = 60, commit probability ρ = 0.25; excursion hazard h_e ∈ {1/90, 1/30}; h_s, η, κ as for the sticky HMM |
| CRIF (P_Z) | `e^S`, `e^E` | D_max = 36, ρ = 0.25; otherwise as CRIF (P_S) |
| W-SGLR | `e^S`, `e^E` | window ∈ {24, 36, 60, 90} scans; τ as for CUSUM |
| BN-GLR | `e^S`, `e^E` | feint bound D ∈ {24, 36} scans; drift δ ∈ {0, 1}; b0: 14 values log-spaced in [1, 120]; b1: −∞ (CUSUM-nested) and 9 values log-spaced in [0.5, 60] |

For the posterior filters, κ (and h_e for CRIF) is fitted for each h_s on the emission-calibration split by
maximizing the mean log posterior of the true intent; η then acts as the threshold. The CRIF structures
(D_max, ρ) were chosen on development flights from D_max ∈ {24, 36, 60, 90} and ρ ∈ {0, 0.25, 0.5, 1}.

End-to-end classifiers (GraphGRU, causal Transformer, bidirectional GRU) drive three rules: argmax change
(no parameter), M-of-N voting with N ∈ {3, 5, 8, 10, 12, 15, 20} and ⌈N/2⌉ ≤ M ≤ N, and CUSUM with the grid
above. They are trained with AdamW (learning rate 3e-4, weight decay 5e-4, batches of 24 flights, gradient
clipping at norm 1, dropout 0.1, hidden size 64) for 40 epochs, keeping the checkpoint with the lowest
validation negative log-likelihood.
