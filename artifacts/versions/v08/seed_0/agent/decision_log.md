# Decision log, seed 0 (dev)

Models: uni2h, virchow2, genbio_pathfm + age and sex. Code 120a6cc0302bdc86399bd3e44cf3f945e48f5275.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with the simplest candidate of each family plus every recipe of the families opened in full at fixed values: E1:raw, L1:raw, P1:raw, N1:ft, M1:ft, X1:raw, A1:raw, C1:raw, C2:raw, K1:raw, K2:raw.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft", "M1:ft", "X1:raw", "A1:raw", "C1:raw", "C2:raw", "K1:raw", "K2:raw"]}`

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

Macro AUROC 0.858 [0.825, 0.889], folds 0.857 ± 0.030; balanced accuracy 0.520 [0.446, 0.593]; 55 s.
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

Macro AUROC 0.849 [0.815, 0.880], folds 0.848 ± 0.041; balanced accuracy 0.473 [0.400, 0.546]; 164 s.
- vs row 5: AUROC gain +0.050 [+0.022, +0.075], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.012 [-0.023, -0.003], wins in 1% of resamples.

## s0-T011 Experiment: K2:raw, screen

Why: Opening: ctx family, simplest candidate (based on s0-D001)

Macro AUROC 0.857 [0.823, 0.888], folds 0.857 ± 0.041; balanced accuracy 0.528 [0.457, 0.597]; 163 s.
- vs row 5: AUROC gain +0.058 [+0.029, +0.082], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.005 [-0.015, +0.006], wins in 18% of resamples.

## s0-D002 Decision: leader

Leading family: clam (best s0-T008 (C1:raw screen, AUROC 0.861)); explore it next.

Based on: s0-T008

- agg: 0.797
- clam: 0.861
- ctx: 0.857
- early: 0.790
- late: 0.801
- learned: 0.770
- mexp: 0.858
- mil: 0.834
- pool: 0.805

Action: `{"family": "clam"}`

## s0-T012 Experiment: C1:ple, screen

Why: Explore the leading family clam: untried age-and-sex encoding ple for C1, the family's best recipe without it (based on s0-D002, s0-T008)

Macro AUROC 0.854 [0.819, 0.886], folds 0.852 ± 0.025; balanced accuracy 0.504 [0.434, 0.572]; 3 s.
- vs row 5: AUROC gain +0.055 [+0.022, +0.083], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.008 [-0.017, +0.002], wins in 5% of resamples.

## s0-T013 Experiment: C2:ple, screen

Why: Explore the leading family clam: untried age-and-sex encoding ple for C2, the family's best recipe without it (based on s0-D002, s0-T008)

Macro AUROC 0.844 [0.806, 0.877], folds 0.842 ± 0.030; balanced accuracy 0.471 [0.399, 0.544]; 2 s.
- vs row 5: AUROC gain +0.045 [+0.018, +0.068], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.018 [-0.031, -0.004], wins in 0% of resamples.

## s0-D003 Decision: leader

Leading family: mexp (best s0-T006 (X1:raw screen, AUROC 0.858)); explore it next.

Based on: s0-T006

- agg: 0.797
- ctx: 0.857
- early: 0.790
- late: 0.801
- learned: 0.770
- mexp: 0.858
- mil: 0.834
- pool: 0.805

Action: `{"family": "mexp"}`

## s0-T014 Experiment: X2:raw, screen

Why: Explore the leading family mexp: untried recipe X2 at the default encoding raw (based on s0-D003, s0-T006)

Macro AUROC 0.855 [0.823, 0.887], folds 0.855 ± 0.033; balanced accuracy 0.536 [0.462, 0.609]; 212 s.
- vs row 5: AUROC gain +0.056 [+0.026, +0.083], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.006 [-0.016, +0.004], wins in 10% of resamples.
- Chosen per fold: [{'weights': [0.2941, 0.4598]}, {'weights': [0.3059, 0.2066]}, {'weights': [0.2825, 0.2675]}, {'weights': [0.3719, 0.4136]}, {'weights': [0.2968, 0.2481]}, {'weights': [0.292, 0.3716]}, {'weights': [0.2823, 0.4777]}, {'weights': [0.3213, 0.3568]}]

## s0-T015 Experiment: X3:raw, screen

Why: Explore the leading family mexp: untried recipe X3 at the default encoding raw (based on s0-D003, s0-T006)

Macro AUROC 0.852 [0.819, 0.883], folds 0.855 ± 0.040; balanced accuracy 0.506 [0.434, 0.577]; 3 s.
- vs row 5: AUROC gain +0.053 [+0.026, +0.077], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.009 [-0.020, +0.002], wins in 4% of resamples.
- Chosen per fold: [{'weights': [0.2661, 0.045, 0.0226, 0.4632]}, {'weights': [0.2752, 0.0731, 0.0, 0.1867]}, {'weights': [0.2559, 0.0426, 0.014, 0.2457]}, {'weights': [0.2581, 0.0942, 0.0479, 0.3783]}, {'weights': [0.2219, 0.0042, 0.092, 0.2656]}, {'weights': [0.2607, 0.0358, 0.03, 0.3601]}, {'weights': [0.3327, 0.0, 0.0046, 0.4424]}, {'weights': [0.2692, 0.08, 0.0076, 0.3406]}]

## s0-T016 Experiment: X4:raw, screen

Why: Explore the leading family mexp: untried recipe X4 at the default encoding raw (based on s0-D003, s0-T006)

Macro AUROC 0.855 [0.822, 0.885], folds 0.857 ± 0.034; balanced accuracy 0.522 [0.446, 0.596]; 2 s.
- vs row 5: AUROC gain +0.055 [+0.028, +0.079], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.007 [-0.017, +0.004], wins in 10% of resamples.
- Chosen per fold: [{'weights': [0.2514, 0.0554, 0.0338, 0.4488]}, {'weights': [0.2855, 0.0862, 0.0, 0.1697]}, {'weights': [0.2527, 0.0545, 0.0182, 0.2198]}, {'weights': [0.2568, 0.0994, 0.0649, 0.4131]}, {'weights': [0.2234, 0.0062, 0.0969, 0.2578]}, {'weights': [0.2555, 0.0408, 0.0401, 0.371]}, {'weights': [0.3363, 0.0, 0.0189, 0.4077]}, {'weights': [0.2608, 0.0874, 0.0246, 0.3384]}]

## s0-T017 Experiment: X1:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X1, the family's best recipe without it (based on s0-D003, s0-T006)

Macro AUROC 0.853 [0.818, 0.885], folds 0.854 ± 0.026; balanced accuracy 0.504 [0.426, 0.583]; 2 s.
- vs row 5: AUROC gain +0.053 [+0.022, +0.081], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.009 [-0.020, +0.002], wins in 5% of resamples.

## s0-D004 Decision: tune_start

Start tuning (the exploration budget is used): the best screened candidate of each recipe, top 3: s0-T003 (P1:raw screen, AUROC 0.805), s0-T002 (L1:raw screen, AUROC 0.801), s0-T001 (E1:raw screen, AUROC 0.790).

Based on: s0-T003, s0-T002, s0-T001

- s0-T001: 0.790
- s0-T002: 0.801
- s0-T003: 0.805

Action: `{"tune": ["P1:raw", "L1:raw", "E1:raw"]}`

## s0-T018 Experiment: P1:raw, tuned

Why: Tune P1:raw: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.802 [0.759, 0.842], folds 0.809 ± 0.030; balanced accuracy 0.431 [0.358, 0.504]; 326 s.
- vs row 5: AUROC gain +0.003 [-0.023, +0.025], wins in 60% of resamples.
- vs best so far (s0-T008): AUROC gain -0.060 [-0.085, -0.034], wins in 0% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'uni2h': 0.001, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'uni2h': 0.01, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.001, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 0.01, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 0.001, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 1.0, 'weights': [1.0, 1.0]}]

## s0-T019 Experiment: L1:raw, tuned

Why: Tune L1:raw: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.796 [0.756, 0.835], folds 0.802 ± 0.051; balanced accuracy 0.429 [0.357, 0.501]; 326 s.
- vs row 5: AUROC gain -0.003 [-0.020, +0.013], wins in 38% of resamples.
- vs best so far (s0-T008): AUROC gain -0.065 [-0.092, -0.039], wins in 0% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T020 Experiment: E1:raw, tuned

Why: Tune E1:raw: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.773 [0.728, 0.818], folds 0.791 ± 0.036; balanced accuracy 0.436 [0.364, 0.508]; 336 s.
- vs row 5: AUROC gain -0.026 [-0.061, +0.005], wins in 6% of resamples.
- vs best so far (s0-T008): AUROC gain -0.088 [-0.119, -0.058], wins in 0% of resamples.
- Chosen per fold: [{'lr__C': 0.001}, {'lr__C': 0.001}, {'lr__C': 1.0}, {'lr__C': 0.01}, {'lr__C': 0.001}, {'lr__C': 0.001}, {'lr__C': 0.001}, {'lr__C': 1.0}]

## s0-D005 Decision: final_selection

s0-T008 (C1:raw screen, AUROC 0.861) has the highest AUROC but does not clearly beat the simpler s0-T006 (X1:raw screen, AUROC 0.858): it wins in 81% of resamples (needs ≥ 90%) and its balanced accuracy differs by +0.005 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T011, s0-T012, s0-T013, s0-T014, s0-T015, s0-T016, s0-T017, s0-T018, s0-T019, s0-T020

- s0-T001: bacc_difference 0.076, candidate E1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.072, mode screen
- s0-T002: bacc_difference 0.100, candidate L1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.060, mode screen
- s0-T003: bacc_difference 0.101, candidate P1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.056, mode screen
- s0-T004: bacc_difference 0.137, candidate N1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.091, mode screen
- s0-T005: bacc_difference 0.017, candidate M1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.027, mode screen
- s0-T006: bacc_difference 0.005, candidate X1:raw, candidate_wins_share 0.812, clearly_beaten False, gain 0.003, mode screen
- s0-T018: bacc_difference 0.094, candidate P1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.060, mode tuned
- s0-T019: bacc_difference 0.095, candidate L1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.065, mode tuned
- s0-T020: bacc_difference 0.089, candidate E1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.088, mode tuned

Action: `{"candidate": "s0-T008", "winner": "s0-T006", "winner_candidate": "X1:raw", "winner_mode": "screen"}`

## s0-A001 Matched baseline 1m: X1:raw, screen

Why: Matched baseline 1m: winner X1:raw (screen) on age and sex alone (based on s0-T006)

Macro AUROC 0.611 [0.560, 0.659], folds 0.634 ± 0.072; balanced accuracy 0.266 [0.208, 0.329]; 2 s.
- vs row 5: AUROC gain -0.188 [-0.252, -0.131], wins in 0% of resamples.
- vs best so far (s0-T008): AUROC gain -0.250 [-0.305, -0.199], wins in 0% of resamples.

## s0-A002 Matched baseline uni2h_m: X1:raw, screen

Why: Matched baseline uni2h_m: winner X1:raw (screen) on uni2h alone (based on s0-T006)

Macro AUROC 0.853 [0.819, 0.885], folds 0.849 ± 0.039; balanced accuracy 0.523 [0.450, 0.594]; 2 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.078], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.009 [-0.022, +0.006], wins in 12% of resamples.

## s0-A003 Matched baseline virchow2_m: X1:raw, screen

Why: Matched baseline virchow2_m: winner X1:raw (screen) on virchow2 alone (based on s0-T006)

Macro AUROC 0.841 [0.802, 0.875], folds 0.837 ± 0.028; balanced accuracy 0.535 [0.461, 0.607]; 2 s.
- vs row 5: AUROC gain +0.042 [+0.003, +0.076], wins in 98% of resamples.
- vs best so far (s0-T008): AUROC gain -0.020 [-0.040, -0.002], wins in 1% of resamples.

## s0-A004 Matched baseline genbio_pathfm_m: X1:raw, screen

Why: Matched baseline genbio_pathfm_m: winner X1:raw (screen) on genbio_pathfm alone (based on s0-T006)

Macro AUROC 0.837 [0.800, 0.873], folds 0.841 ± 0.027; balanced accuracy 0.475 [0.401, 0.548]; 2 s.
- vs row 5: AUROC gain +0.038 [+0.004, +0.070], wins in 99% of resamples.
- vs best so far (s0-T008): AUROC gain -0.024 [-0.040, -0.009], wins in 0% of resamples.

## s0-A005 Matched baseline diag_no_meta: X1:raw, screen

Why: Matched baseline diag_no_meta: winner X1:raw (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T006)

Macro AUROC 0.853 [0.819, 0.885], folds 0.856 ± 0.025; balanced accuracy 0.522 [0.451, 0.595]; 2 s.
- vs row 5: AUROC gain +0.054 [+0.023, +0.082], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.008 [-0.018, +0.002], wins in 5% of resamples.

## s0-A006 Matched baseline 5m: X1:raw, screen

Why: Matched row 5: winner X1:raw (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T006)

Macro AUROC 0.856 [0.823, 0.888], folds 0.855 ± 0.042; balanced accuracy 0.496 [0.425, 0.569]; 2 s.
- vs row 5: AUROC gain +0.057 [+0.030, +0.083], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain -0.005 [-0.017, +0.007], wins in 21% of resamples.
