# Decision log, seed 0 (dev)

Models: uni2h, virchow2, genbio_pathfm + age and sex. Code 57fd20b7766587e76bc04846c92cd30e48855f98.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with one candidate per family at fixed values: E1:raw, L1:raw, P1:raw, N1:ft, M1:ft, X1:raw, A1:raw.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft", "M1:ft", "X1:raw", "A1:raw"]}`

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

Macro AUROC 0.834 [0.794, 0.872], folds 0.824 ± 0.030; balanced accuracy 0.508 [0.433, 0.581]; 183 s.
- vs row 5: AUROC gain +0.035 [-0.001, +0.066], wins in 97% of resamples.
- vs best so far (s0-T003): AUROC gain +0.029 [-0.001, +0.057], wins in 96% of resamples.

## s0-T006 Experiment: X1:raw, screen

Why: Opening: mexp family, simplest candidate (based on s0-D001)

Macro AUROC 0.858 [0.825, 0.889], folds 0.857 ± 0.030; balanced accuracy 0.520 [0.446, 0.593]; 59 s.
- vs row 5: AUROC gain +0.059 [+0.029, +0.085], wins in 100% of resamples.
- vs best so far (s0-T005): AUROC gain +0.024 [+0.009, +0.041], wins in 100% of resamples.

## s0-T007 Experiment: A1:raw, screen

Why: Opening: agg family, simplest candidate (based on s0-D001)

Macro AUROC 0.797 [0.757, 0.834], folds 0.799 ± 0.058; balanced accuracy 0.453 [0.380, 0.525]; 210 s.
- vs row 5: AUROC gain -0.002 [-0.034, +0.027], wins in 44% of resamples.
- vs best so far (s0-T006): AUROC gain -0.061 [-0.094, -0.030], wins in 0% of resamples.

## s0-D002 Decision: leader

Leading family: mexp (best s0-T006 (X1:raw screen, AUROC 0.858)); explore it next.

Based on: s0-T006

- agg: 0.797
- early: 0.790
- late: 0.801
- learned: 0.770
- mexp: 0.858
- mil: 0.834
- pool: 0.805

Action: `{"family": "mexp"}`

## s0-T008 Experiment: X2:raw, screen

Why: Explore the leading family mexp: untried recipe X2 at the default encoding raw (based on s0-D002, s0-T006)

Macro AUROC 0.855 [0.823, 0.887], folds 0.855 ± 0.033; balanced accuracy 0.536 [0.462, 0.609]; 529 s.
- vs row 5: AUROC gain +0.056 [+0.026, +0.083], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.003 [-0.009, +0.004], wins in 23% of resamples.
- Chosen per fold: [{'weights': [0.2941, 0.4598]}, {'weights': [0.3059, 0.2066]}, {'weights': [0.2825, 0.2675]}, {'weights': [0.3719, 0.4136]}, {'weights': [0.2968, 0.2481]}, {'weights': [0.292, 0.3716]}, {'weights': [0.2823, 0.4777]}, {'weights': [0.3213, 0.3568]}]

## s0-T009 Experiment: X3:raw, screen

Why: Explore the leading family mexp: untried recipe X3 at the default encoding raw (based on s0-D002, s0-T006)

Macro AUROC 0.852 [0.819, 0.883], folds 0.855 ± 0.040; balanced accuracy 0.506 [0.434, 0.577]; 3 s.
- vs row 5: AUROC gain +0.053 [+0.026, +0.077], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.006 [-0.015, +0.004], wins in 13% of resamples.
- Chosen per fold: [{'weights': [0.2661, 0.045, 0.0226, 0.4632]}, {'weights': [0.2752, 0.0731, 0.0, 0.1867]}, {'weights': [0.2559, 0.0426, 0.014, 0.2457]}, {'weights': [0.2581, 0.0942, 0.0479, 0.3783]}, {'weights': [0.2219, 0.0042, 0.092, 0.2656]}, {'weights': [0.2607, 0.0358, 0.03, 0.3601]}, {'weights': [0.3327, 0.0, 0.0046, 0.4424]}, {'weights': [0.2692, 0.08, 0.0076, 0.3406]}]

## s0-T010 Experiment: X4:raw, screen

Why: Explore the leading family mexp: untried recipe X4 at the default encoding raw (based on s0-D002, s0-T006)

Macro AUROC 0.855 [0.822, 0.885], folds 0.857 ± 0.034; balanced accuracy 0.522 [0.446, 0.596]; 2 s.
- vs row 5: AUROC gain +0.055 [+0.028, +0.079], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.003 [-0.013, +0.006], wins in 26% of resamples.
- Chosen per fold: [{'weights': [0.2514, 0.0554, 0.0338, 0.4488]}, {'weights': [0.2855, 0.0862, 0.0, 0.1697]}, {'weights': [0.2527, 0.0545, 0.0182, 0.2198]}, {'weights': [0.2568, 0.0994, 0.0649, 0.4131]}, {'weights': [0.2234, 0.0062, 0.0969, 0.2578]}, {'weights': [0.2555, 0.0408, 0.0401, 0.371]}, {'weights': [0.3363, 0.0, 0.0189, 0.4077]}, {'weights': [0.2608, 0.0874, 0.0246, 0.3384]}]

## s0-T011 Experiment: X1:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X1, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.853 [0.818, 0.885], folds 0.854 ± 0.026; balanced accuracy 0.504 [0.426, 0.583]; 2 s.
- vs row 5: AUROC gain +0.053 [+0.022, +0.081], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.005 [-0.013, +0.002], wins in 6% of resamples.

## s0-T012 Experiment: X2:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X2, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.854 [0.820, 0.885], folds 0.856 ± 0.027; balanced accuracy 0.522 [0.447, 0.598]; 3 s.
- vs row 5: AUROC gain +0.055 [+0.023, +0.082], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.004 [-0.011, +0.003], wins in 13% of resamples.
- Chosen per fold: [{'weights': [0.298, 0.1182]}, {'weights': [0.3084, 0.0]}, {'weights': [0.2872, 0.0602]}, {'weights': [0.3733, 0.0]}, {'weights': [0.2995, 0.0258]}, {'weights': [0.3009, 0.0]}, {'weights': [0.2871, 0.0588]}, {'weights': [0.3257, 0.0745]}]

## s0-T013 Experiment: X4:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X4, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.854 [0.821, 0.884], folds 0.857 ± 0.033; balanced accuracy 0.527 [0.453, 0.602]; 2 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.079], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.004 [-0.014, +0.006], wins in 22% of resamples.
- Chosen per fold: [{'weights': [0.2547, 0.0543, 0.0348, 0.1432]}, {'weights': [0.2875, 0.0865, 0.0, 0.0]}, {'weights': [0.2562, 0.0581, 0.0155, 0.0698]}, {'weights': [0.2723, 0.0918, 0.0595, 0.0]}, {'weights': [0.226, 0.0075, 0.0953, 0.0689]}, {'weights': [0.2638, 0.0585, 0.0242, 0.0034]}, {'weights': [0.3434, 0.0, 0.0181, 0.074]}, {'weights': [0.2616, 0.0906, 0.0244, 0.0919]}]

## s0-T014 Experiment: X3:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X3, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.851 [0.817, 0.882], folds 0.851 ± 0.037; balanced accuracy 0.502 [0.428, 0.573]; 2 s.
- vs row 5: AUROC gain +0.052 [+0.023, +0.077], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.007 [-0.017, +0.003], wins in 10% of resamples.
- Chosen per fold: [{'weights': [0.2681, 0.0453, 0.0235, 0.1221]}, {'weights': [0.2764, 0.0741, 0.0, 0.0]}, {'weights': [0.2601, 0.0479, 0.0099, 0.0532]}, {'weights': [0.272, 0.0896, 0.0425, 0.0]}, {'weights': [0.2223, 0.0083, 0.0898, 0.048]}, {'weights': [0.2674, 0.0553, 0.015, 0.0]}, {'weights': [0.3413, 0.0, 0.0021, 0.0423]}, {'weights': [0.272, 0.0844, 0.005, 0.0586]}]

## s0-D003 Decision: leader

Leading family: mil (best s0-T005 (M1:ft screen, AUROC 0.834)); explore it next.

Based on: s0-T005

- agg: 0.797
- early: 0.790
- late: 0.801
- learned: 0.770
- mil: 0.834
- pool: 0.805

Action: `{"family": "mil"}`

## s0-T015 Experiment: M2:ft, screen

Why: Explore the leading family mil: untried recipe M2 at the default encoding ft (based on s0-D003, s0-T005)

Macro AUROC 0.851 [0.813, 0.886], folds 0.848 ± 0.033; balanced accuracy 0.502 [0.431, 0.575]; 184 s.
- vs row 5: AUROC gain +0.052 [+0.014, +0.084], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.007 [-0.024, +0.006], wins in 17% of resamples.

## s0-T016 Experiment: M3:ft, screen

Why: Explore the leading family mil: untried recipe M3 at the default encoding ft (based on s0-D003, s0-T015)

Macro AUROC 0.825 [0.786, 0.864], folds 0.817 ± 0.029; balanced accuracy 0.492 [0.418, 0.565]; 190 s.
- vs row 5: AUROC gain +0.026 [-0.009, +0.056], wins in 94% of resamples.
- vs best so far (s0-T006): AUROC gain -0.032 [-0.050, -0.018], wins in 0% of resamples.

## s0-T017 Experiment: M2:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M2, the family's best recipe without it (based on s0-D003, s0-T015)

Macro AUROC 0.851 [0.813, 0.885], folds 0.847 ± 0.029; balanced accuracy 0.511 [0.438, 0.585]; 185 s.
- vs row 5: AUROC gain +0.051 [+0.014, +0.083], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.007 [-0.023, +0.006], wins in 15% of resamples.

## s0-D004 Decision: tune_start

Start tuning (the exploration budget is used): the best screened candidate of each recipe, top 3: s0-T015 (M2:ft screen, AUROC 0.851), s0-T005 (M1:ft screen, AUROC 0.834), s0-T016 (M3:ft screen, AUROC 0.825).

Based on: s0-T015, s0-T005, s0-T016

- s0-T005: 0.834
- s0-T015: 0.851
- s0-T016: 0.825

Action: `{"tune": ["M2:ft", "M1:ft", "M3:ft"]}`

## s0-T018 Experiment: M2:ft, tuned

Why: Tune M2:ft: C or weight decay and size by inner CV (based on s0-D004)

Status: **timed_out** after 1802 s.

## s0-T019 Experiment: M1:ft, tuned

Why: Tune M1:ft: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.836 [0.796, 0.874], folds 0.825 ± 0.032; balanced accuracy 0.511 [0.439, 0.583]; 1588 s.
- vs row 5: AUROC gain +0.037 [+0.000, +0.069], wins in 98% of resamples.
- vs best so far (s0-T006): AUROC gain -0.022 [-0.039, -0.007], wins in 0% of resamples.
- Chosen per fold: [{'weight_decay': 0.0001}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}]

## s0-T020 Experiment: M3:ft, tuned

Why: Tune M3:ft: C or weight decay and size by inner CV (based on s0-D004)

Status: **timed_out** after 1802 s.

## s0-D005 Decision: final_selection

s0-T006 (X1:raw screen, AUROC 0.858) has the highest AUROC but does not clearly beat the simpler s0-T015 (M2:ft screen, AUROC 0.851): it wins in 83% of resamples (needs ≥ 90%) and its balanced accuracy differs by +0.018 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T011, s0-T012, s0-T013, s0-T014, s0-T015, s0-T016, s0-T017, s0-T019

- s0-T001: bacc_difference 0.072, candidate E1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.068, mode screen
- s0-T002: bacc_difference 0.095, candidate L1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.057, mode screen
- s0-T003: bacc_difference 0.096, candidate P1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.053, mode screen
- s0-T004: bacc_difference 0.132, candidate N1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.088, mode screen
- s0-T005: bacc_difference 0.012, candidate M1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.024, mode screen
- s0-T015: bacc_difference 0.018, candidate M2:ft, candidate_wins_share 0.828, clearly_beaten False, gain 0.007, mode screen
- s0-T019: bacc_difference 0.009, candidate M1:ft, candidate_wins_share 0.999, clearly_beaten True, gain 0.022, mode tuned

Action: `{"candidate": "s0-T006", "winner": "s0-T015", "winner_candidate": "M2:ft", "winner_mode": "screen"}`

## s0-A001 Matched baseline uni2h_m: M2:ft, screen

Why: Matched baseline uni2h_m: winner M2:ft (screen) on uni2h alone (based on s0-T015)

Macro AUROC 0.853 [0.819, 0.885], folds 0.849 ± 0.039; balanced accuracy 0.523 [0.450, 0.594]; 19 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.078], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.005 [-0.019, +0.009], wins in 23% of resamples.

## s0-A002 Matched baseline virchow2_m: M2:ft, screen

Why: Matched baseline virchow2_m: winner M2:ft (screen) on virchow2 alone (based on s0-T015)

Macro AUROC 0.841 [0.802, 0.875], folds 0.837 ± 0.028; balanced accuracy 0.535 [0.461, 0.607]; 26 s.
- vs row 5: AUROC gain +0.042 [+0.003, +0.076], wins in 98% of resamples.
- vs best so far (s0-T006): AUROC gain -0.017 [-0.034, -0.002], wins in 2% of resamples.

## s0-A003 Matched baseline genbio_pathfm_m: M2:ft, screen

Why: Matched baseline genbio_pathfm_m: winner M2:ft (screen) on genbio_pathfm alone (based on s0-T015)

Macro AUROC 0.837 [0.800, 0.873], folds 0.841 ± 0.027; balanced accuracy 0.475 [0.401, 0.548]; 29 s.
- vs row 5: AUROC gain +0.038 [+0.004, +0.070], wins in 99% of resamples.
- vs best so far (s0-T006): AUROC gain -0.021 [-0.034, -0.008], wins in 0% of resamples.

## s0-A004 Matched baseline diag_no_meta: M2:ft, screen

Why: Matched baseline diag_no_meta: winner M2:ft (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T015)

Macro AUROC 0.853 [0.816, 0.888], folds 0.850 ± 0.032; balanced accuracy 0.518 [0.444, 0.592]; 176 s.
- vs row 5: AUROC gain +0.054 [+0.017, +0.086], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.004 [-0.020, +0.008], wins in 27% of resamples.

## s0-A005 Matched baseline 5m: M2:ft, screen

Why: Matched row 5: winner M2:ft (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T015)

Macro AUROC 0.851 [0.817, 0.882], folds 0.847 ± 0.043; balanced accuracy 0.512 [0.439, 0.581]; 21 s.
- vs row 5: AUROC gain +0.052 [+0.025, +0.077], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.007 [-0.021, +0.007], wins in 16% of resamples.
