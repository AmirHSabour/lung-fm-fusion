# Decision log, seed 0 (dev)

Models: uni2h, virchow2, genbio_pathfm + age and sex. Code 4afbf67439d27f35ff05538416c23b37c95dc45a.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with the simplest candidate of each family plus every recipe of the families opened in full at fixed values: E1:raw, L1:raw, P1:raw, N1:ft, M1:ft, X1:raw, A1:raw, C1:raw, C2:raw, K1:raw, K2:raw, Z1:raw, Z2:raw, Q1:raw, Q2:ft, H1:raw, H2:raw.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft", "M1:ft", "X1:raw", "A1:raw", "C1:raw", "C2:raw", "K1:raw", "K2:raw", "Z1:raw", "Z2:raw", "Q1:raw", "Q2:ft", "H1:raw", "H2:raw"]}`

## s0-T001 Experiment: E1:raw, screen

Why: Opening: early family, simplest candidate (based on s0-D001)

Macro AUROC 0.790 [0.744, 0.831], folds 0.802 ± 0.052; balanced accuracy 0.448 [0.374, 0.524]; 10 s.
- vs row 5: AUROC gain -0.010 [-0.035, +0.013], wins in 20% of resamples.

## s0-T002 Experiment: L1:raw, screen

Why: Opening: late family, simplest candidate (based on s0-D001)

Macro AUROC 0.801 [0.760, 0.839], folds 0.810 ± 0.051; balanced accuracy 0.425 [0.351, 0.496]; 7 s.
- vs row 5: AUROC gain +0.002 [-0.020, +0.021], wins in 57% of resamples.
- vs best so far (s0-T001): AUROC gain +0.012 [-0.002, +0.026], wins in 95% of resamples.

## s0-T003 Experiment: P1:raw, screen

Why: Opening: pool family, simplest candidate (based on s0-D001)

Macro AUROC 0.805 [0.764, 0.842], folds 0.813 ± 0.053; balanced accuracy 0.423 [0.352, 0.502]; 7 s.
- vs row 5: AUROC gain +0.006 [-0.017, +0.026], wins in 70% of resamples.
- vs best so far (s0-T002): AUROC gain +0.004 [-0.006, +0.014], wins in 78% of resamples.

## s0-T004 Experiment: N1:ft, screen

Why: Opening: learned family, simplest candidate (based on s0-D001)

Macro AUROC 0.770 [0.727, 0.809], folds 0.755 ± 0.046; balanced accuracy 0.388 [0.315, 0.458]; 7 s.
- vs row 5: AUROC gain -0.029 [-0.061, +0.001], wins in 3% of resamples.
- vs best so far (s0-T003): AUROC gain -0.035 [-0.065, -0.006], wins in 1% of resamples.

## s0-T005 Experiment: M1:ft, screen

Why: Opening: mil family, simplest candidate (based on s0-D001)

Macro AUROC 0.834 [0.794, 0.872], folds 0.824 ± 0.030; balanced accuracy 0.508 [0.433, 0.581]; 51 s.
- vs row 5: AUROC gain +0.035 [-0.001, +0.066], wins in 97% of resamples.
- vs best so far (s0-T003): AUROC gain +0.029 [-0.001, +0.057], wins in 96% of resamples.

## s0-T006 Experiment: X1:raw, screen

Why: Opening: mexp family, simplest candidate (based on s0-D001)

Macro AUROC 0.858 [0.825, 0.889], folds 0.857 ± 0.030; balanced accuracy 0.520 [0.446, 0.593]; 54 s.
- vs row 5: AUROC gain +0.059 [+0.029, +0.085], wins in 100% of resamples.
- vs best so far (s0-T005): AUROC gain +0.024 [+0.009, +0.041], wins in 100% of resamples.

## s0-T007 Experiment: A1:raw, screen

Why: Opening: agg family, simplest candidate (based on s0-D001)

Macro AUROC 0.797 [0.757, 0.834], folds 0.799 ± 0.058; balanced accuracy 0.453 [0.380, 0.525]; 53 s.
- vs row 5: AUROC gain -0.002 [-0.034, +0.027], wins in 44% of resamples.
- vs best so far (s0-T006): AUROC gain -0.061 [-0.094, -0.030], wins in 0% of resamples.

## s0-T008 Experiment: C1:raw, screen

Why: Opening: clam family, simplest candidate (based on s0-D001)

Macro AUROC 0.861 [0.828, 0.892], folds 0.858 ± 0.030; balanced accuracy 0.524 [0.454, 0.593]; 63 s.
- vs row 5: AUROC gain +0.062 [+0.033, +0.088], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain +0.003 [-0.004, +0.011], wins in 81% of resamples.

## s0-T009 Experiment: C2:raw, screen

Why: Opening: clam family, simplest candidate (based on s0-D001)

Macro AUROC 0.852 [0.817, 0.884], folds 0.855 ± 0.041; balanced accuracy 0.469 [0.401, 0.538]; 63 s.
- vs row 5: AUROC gain +0.053 [+0.028, +0.076], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.009 [-0.020, +0.002], wins in 6% of resamples.

## s0-T010 Experiment: K1:raw, screen

Why: Opening: ctx family, simplest candidate (based on s0-D001)

Macro AUROC 0.849 [0.815, 0.880], folds 0.848 ± 0.041; balanced accuracy 0.473 [0.400, 0.546]; 155 s.
- vs row 5: AUROC gain +0.050 [+0.022, +0.075], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.012 [-0.023, -0.003], wins in 1% of resamples.

## s0-T011 Experiment: K2:raw, screen

Why: Opening: ctx family, simplest candidate (based on s0-D001)

Macro AUROC 0.857 [0.823, 0.888], folds 0.857 ± 0.041; balanced accuracy 0.528 [0.457, 0.597]; 165 s.
- vs row 5: AUROC gain +0.058 [+0.029, +0.082], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.005 [-0.015, +0.006], wins in 18% of resamples.

## s0-T012 Experiment: Z1:raw, screen

Why: Opening: ens family, simplest candidate (based on s0-D001)

Macro AUROC 0.861 [0.828, 0.891], folds 0.861 ± 0.031; balanced accuracy 0.530 [0.457, 0.600]; 3 s.
- vs row 5: AUROC gain +0.061 [+0.033, +0.088], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.001 [-0.006, +0.004], wins in 38% of resamples.

## s0-T013 Experiment: Z2:raw, screen

Why: Opening: ens family, simplest candidate (based on s0-D001)

Macro AUROC 0.858 [0.825, 0.889], folds 0.861 ± 0.033; balanced accuracy 0.512 [0.441, 0.584]; 2 s.
- vs row 5: AUROC gain +0.059 [+0.030, +0.085], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.003 [-0.010, +0.003], wins in 15% of resamples.

## s0-T014 Experiment: Q1:raw, screen

Why: Opening: xq family, simplest candidate (based on s0-D001)

Macro AUROC 0.847 [0.809, 0.882], folds 0.850 ± 0.037; balanced accuracy 0.523 [0.451, 0.597]; 56 s.
- vs row 5: AUROC gain +0.048 [+0.018, +0.075], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.014 [-0.032, +0.004], wins in 7% of resamples.

## s0-T015 Experiment: Q2:ft, screen

Why: Opening: xqm family, simplest candidate (based on s0-D001)

Macro AUROC 0.846 [0.810, 0.880], folds 0.849 ± 0.025; balanced accuracy 0.520 [0.445, 0.596]; 57 s.
- vs row 5: AUROC gain +0.047 [+0.014, +0.075], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.016 [-0.034, +0.004], wins in 6% of resamples.

## s0-T016 Experiment: H1:raw, screen

Why: Opening: hier family, simplest candidate (based on s0-D001)

Macro AUROC 0.863 [0.831, 0.894], folds 0.863 ± 0.033; balanced accuracy 0.521 [0.450, 0.592]; 55 s.
- vs row 5: AUROC gain +0.064 [+0.033, +0.093], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain +0.002 [-0.005, +0.009], wins in 71% of resamples.

## s0-T017 Experiment: H2:raw, screen

Why: Opening: hier family, simplest candidate (based on s0-D001)

Macro AUROC 0.866 [0.833, 0.896], folds 0.866 ± 0.028; balanced accuracy 0.529 [0.457, 0.600]; 64 s.
- vs row 5: AUROC gain +0.066 [+0.035, +0.094], wins in 100% of resamples.
- vs best so far (s0-T016): AUROC gain +0.002 [-0.004, +0.009], wins in 75% of resamples.

## s0-D002 Decision: drop

Drop clam: the best experiment s0-T017 (H2:raw screen, AUROC 0.866) beats the clam family's best s0-T008 (C1:raw screen, AUROC 0.861) in 92% of resamples (≥ 90%).

Based on: s0-T017, s0-T008

- best_wins_share: 0.920
- gain: 0.004

Action: `{"family": "clam"}`

## s0-D003 Decision: drop

Drop ctx: the best experiment s0-T017 (H2:raw screen, AUROC 0.866) beats the ctx family's best s0-T011 (K2:raw screen, AUROC 0.857) in 95% of resamples (≥ 90%).

Based on: s0-T017, s0-T011

- best_wins_share: 0.949
- gain: 0.009

Action: `{"family": "ctx"}`

## s0-D004 Decision: drop

Drop ens: the best experiment s0-T017 (H2:raw screen, AUROC 0.866) beats the ens family's best s0-T012 (Z1:raw screen, AUROC 0.861) in 93% of resamples (≥ 90%).

Based on: s0-T017, s0-T012

- best_wins_share: 0.931
- gain: 0.005

Action: `{"family": "ens"}`

## s0-D005 Decision: tune_start

Start tuning (the exploration budget is used): the best screened candidate of each recipe, top 3: s0-T003 (P1:raw screen, AUROC 0.805), s0-T002 (L1:raw screen, AUROC 0.801), s0-T001 (E1:raw screen, AUROC 0.790).

Based on: s0-T003, s0-T002, s0-T001

- s0-T001: 0.790
- s0-T002: 0.801
- s0-T003: 0.805

Action: `{"tune": ["P1:raw", "L1:raw", "E1:raw"]}`

## s0-T018 Experiment: P1:raw, tuned

Why: Tune P1:raw: C or weight decay and size by inner CV (based on s0-D005)

Macro AUROC 0.802 [0.759, 0.842], folds 0.809 ± 0.030; balanced accuracy 0.431 [0.358, 0.504]; 327 s.
- vs row 5: AUROC gain +0.003 [-0.023, +0.025], wins in 60% of resamples.
- vs best so far (s0-T017): AUROC gain -0.064 [-0.091, -0.036], wins in 0% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'uni2h': 0.001, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'uni2h': 0.01, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.001, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 0.01, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 0.001, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 1.0, 'weights': [1.0, 1.0]}]

## s0-T019 Experiment: L1:raw, tuned

Why: Tune L1:raw: C or weight decay and size by inner CV (based on s0-D005)

Macro AUROC 0.796 [0.756, 0.835], folds 0.802 ± 0.051; balanced accuracy 0.429 [0.357, 0.501]; 329 s.
- vs row 5: AUROC gain -0.003 [-0.020, +0.013], wins in 38% of resamples.
- vs best so far (s0-T017): AUROC gain -0.069 [-0.097, -0.042], wins in 0% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T020 Experiment: E1:raw, tuned

Why: Tune E1:raw: C or weight decay and size by inner CV (based on s0-D005)

Macro AUROC 0.773 [0.728, 0.818], folds 0.791 ± 0.036; balanced accuracy 0.436 [0.364, 0.508]; 635 s.
- vs row 5: AUROC gain -0.026 [-0.061, +0.005], wins in 6% of resamples.
- vs best so far (s0-T017): AUROC gain -0.092 [-0.123, -0.062], wins in 0% of resamples.
- Chosen per fold: [{'lr__C': 0.001}, {'lr__C': 0.001}, {'lr__C': 1.0}, {'lr__C': 0.01}, {'lr__C': 0.001}, {'lr__C': 0.001}, {'lr__C': 0.001}, {'lr__C': 1.0}]

## s0-D006 Decision: final_selection

s0-T017 (H2:raw screen, AUROC 0.866) has the highest AUROC but does not clearly beat the simpler s0-T016 (H1:raw screen, AUROC 0.863): it wins in 75% of resamples (needs ≥ 90%) and its balanced accuracy differs by +0.008 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T014, s0-T015, s0-T016, s0-T017, s0-T018, s0-T019, s0-T020

- s0-T001: bacc_difference 0.081, candidate E1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.076, mode screen
- s0-T002: bacc_difference 0.104, candidate L1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.065, mode screen
- s0-T003: bacc_difference 0.106, candidate P1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.061, mode screen
- s0-T004: bacc_difference 0.142, candidate N1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.095, mode screen
- s0-T005: bacc_difference 0.021, candidate M1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.031, mode screen
- s0-T006: bacc_difference 0.009, candidate X1:raw, candidate_wins_share 0.965, clearly_beaten True, gain 0.008, mode screen
- s0-T007: bacc_difference 0.076, candidate A1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.068, mode screen
- s0-T014: bacc_difference 0.006, candidate Q1:raw, candidate_wins_share 0.976, clearly_beaten True, gain 0.018, mode screen
- s0-T015: bacc_difference 0.009, candidate Q2:ft, candidate_wins_share 0.977, clearly_beaten True, gain 0.020, mode screen
- s0-T016: bacc_difference 0.008, candidate H1:raw, candidate_wins_share 0.747, clearly_beaten False, gain 0.002, mode screen
- s0-T018: bacc_difference 0.098, candidate P1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.064, mode tuned
- s0-T019: bacc_difference 0.100, candidate L1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.069, mode tuned
- s0-T020: bacc_difference 0.093, candidate E1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.092, mode tuned

Action: `{"candidate": "s0-T017", "winner": "s0-T016", "winner_candidate": "H1:raw", "winner_mode": "screen"}`

## s0-A001 Matched baseline 1m: H1:raw, screen

Why: Matched baseline 1m: winner H1:raw (screen) on age and sex alone (based on s0-T016)

Macro AUROC 0.611 [0.560, 0.659], folds 0.634 ± 0.072; balanced accuracy 0.266 [0.208, 0.329]; 2 s.
- vs row 5: AUROC gain -0.188 [-0.252, -0.131], wins in 0% of resamples.
- vs best so far (s0-T017): AUROC gain -0.255 [-0.308, -0.204], wins in 0% of resamples.

## s0-A002 Matched baseline uni2h_m: H1:raw, screen

Why: Matched baseline uni2h_m: winner H1:raw (screen) on uni2h alone (based on s0-T016)

Macro AUROC 0.856 [0.825, 0.888], folds 0.851 ± 0.038; balanced accuracy 0.517 [0.443, 0.587]; 3 s.
- vs row 5: AUROC gain +0.057 [+0.028, +0.086], wins in 100% of resamples.
- vs best so far (s0-T017): AUROC gain -0.009 [-0.023, +0.007], wins in 12% of resamples.

## s0-A003 Matched baseline virchow2_m: H1:raw, screen

Why: Matched baseline virchow2_m: winner H1:raw (screen) on virchow2 alone (based on s0-T016)

Macro AUROC 0.854 [0.819, 0.887], folds 0.851 ± 0.031; balanced accuracy 0.542 [0.470, 0.612]; 2 s.
- vs row 5: AUROC gain +0.055 [+0.017, +0.088], wins in 100% of resamples.
- vs best so far (s0-T017): AUROC gain -0.012 [-0.028, +0.004], wins in 7% of resamples.

## s0-A004 Matched baseline genbio_pathfm_m: H1:raw, screen

Why: Matched baseline genbio_pathfm_m: winner H1:raw (screen) on genbio_pathfm alone (based on s0-T016)

Macro AUROC 0.841 [0.802, 0.877], folds 0.837 ± 0.035; balanced accuracy 0.465 [0.391, 0.542]; 3 s.
- vs row 5: AUROC gain +0.042 [+0.006, +0.074], wins in 99% of resamples.
- vs best so far (s0-T017): AUROC gain -0.024 [-0.042, -0.008], wins in 0% of resamples.

## s0-A005 Matched baseline diag_no_meta: H1:raw, screen

Why: Matched baseline diag_no_meta: winner H1:raw (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T016)

Macro AUROC 0.860 [0.827, 0.891], folds 0.858 ± 0.025; balanced accuracy 0.528 [0.456, 0.598]; 2 s.
- vs row 5: AUROC gain +0.061 [+0.028, +0.090], wins in 100% of resamples.
- vs best so far (s0-T017): AUROC gain -0.006 [-0.015, +0.004], wins in 12% of resamples.

## s0-A006 Matched baseline 5m: H1:raw, screen

Why: Matched row 5: winner H1:raw (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T016)

Macro AUROC 0.859 [0.827, 0.889], folds 0.853 ± 0.041; balanced accuracy 0.532 [0.461, 0.604]; 2 s.
- vs row 5: AUROC gain +0.060 [+0.031, +0.087], wins in 100% of resamples.
- vs best so far (s0-T017): AUROC gain -0.007 [-0.019, +0.006], wins in 14% of resamples.
