# final_tf2k_plan_v1 — Transformer excursion evidence + BN-GLR on fresh test flights

Status: written before any tf2k model is trained and before the fresh test flights are generated.
Approved by the author in chat on 2026-10-16 ("run the Transformer evidence + BN-GLR experiment on new test seeds").

## Question
final_e2e_v1 found that a causal Transformer trained on K-class labels, decided by CUSUM, has a lower
feint false-alarm rate than TGE evidence + BN-GLR (degraded radar, 24 s). Is that gap due to the
encoder (sequence model with unbounded memory) or to the decision rule? We give the same Transformer the
2K-class head (intent x steady/excursion) and run the decision rules on its evidence.

## Encoders (model seeds 42, 123, 456, 789, 1024)
- TGE-2K: the frozen tge-v2 encoder of final_test_v1 (unchanged).
- TF-K: the frozen causal Transformer of final_e2e_v1 (unchanged).
- TF-2K (new): models/E2E.py kind 'tf', 16 outputs, trained on train_v2 with the same 2K labels as
  tge-v2 (scripts/stage3_learned.labels), the final_e2e_v1 recipe (40 epochs, best encoder_val_v2 NLL).
  Temperature on emission_calibration_v2; prior from train_v2 2K label counts; evidence as for TGE
  (scripts/stage6_sensitivity.evidence: marginal and split).

## Decision rules and operating points
- TGE-2K: Sticky-marginal, CUSUM-marginal, BN-GLR at the frozen final_test_v1 points (no reselection).
- TF-K: CUSUM, M-of-N at the frozen final_e2e_v1 points (no reselection).
- TF-2K: Sticky-marginal, CUSUM-marginal, BN-GLR (same grids as final_test_v1), selected by the
  final_test_plan_v1 rule: blind-feasible on alarm_threshold_v2_x3 (steady/switch, exact 95% bound
  <= 0.05), development censored switch delay <= budget, lowest development feint FA (ties: lower delay).
  Budgets: degraded 24 and 26 s, nominal 22 and 24 s. Frozen with SHA-256 before the fresh test.

## Fresh test flights
Never generated before: split names final_test_fresh_{degraded,nominal}_v3, test_plan of
configs/ast_v2.json (200 steady, 240 switch, 200 feint, 120 feint_switch per radar and seed).
Generated once, after the selection is frozen. The opened final_test_v2 flights are not used.

## Hypotheses (degraded radar, 24 s; feint flights; any false alarm)
- H1 (primary): TF-2K + BN-GLR vs TF-2K + CUSUM (does the rule matter on Transformer evidence?).
- H2: TF-2K + BN-GLR vs TF-K + CUSUM (does excursion supervision help the Transformer?).
- H3: TF-2K + BN-GLR vs TGE-2K + BN-GLR (does the sequence model help BN-GLR?).
Two-sided stratified exact McNemar pooled over seeds, Holm over H1-H3; flight bootstrap 95% CI of
the difference (4000 resamples). Secondary (no correction): the same contrasts at 26 s and on nominal
radar; replication of TGE-2K BN-GLR vs Sticky and vs CUSUM on fresh flights; switch delay, detection,
feint->switch detection. Results are reported whatever they are.
