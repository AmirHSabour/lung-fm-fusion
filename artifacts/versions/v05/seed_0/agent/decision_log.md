# Decision log, seed 0 (dev)

Models: uni2h, virchow2 + age and sex. Code 69ca764cf1fa03f9510da3472aebb054772bd2b2.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with one candidate per family at fixed values: E1:raw, L1:raw, P1:raw, N1:ft, M1:ft, X1:raw.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft", "M1:ft", "X1:raw"]}`

## s0-T001 Experiment: E1:raw, screen

Why: Opening: early family, simplest candidate (based on s0-D001)

Macro AUROC 0.786 [0.740, 0.828], folds 0.796 ± 0.056; balanced accuracy 0.435 [0.362, 0.507]; 5 s.
- vs row 5: AUROC gain -0.013 [-0.039, +0.009], wins in 12% of resamples.

## s0-T002 Experiment: L1:raw, screen

Why: Opening: late family, simplest candidate (based on s0-D001)

Macro AUROC 0.799 [0.757, 0.838], folds 0.808 ± 0.057; balanced accuracy 0.414 [0.345, 0.485]; 4 s.
- vs row 5: AUROC gain -0.000 [-0.023, +0.020], wins in 49% of resamples.
- vs best so far (s0-T001): AUROC gain +0.013 [-0.001, +0.028], wins in 97% of resamples.

## s0-T003 Experiment: P1:raw, screen

Why: Opening: pool family, simplest candidate (based on s0-D001)

Macro AUROC 0.807 [0.766, 0.844], folds 0.811 ± 0.060; balanced accuracy 0.423 [0.350, 0.497]; 4 s.
- vs row 5: AUROC gain +0.008 [-0.015, +0.027], wins in 76% of resamples.
- vs best so far (s0-T002): AUROC gain +0.008 [-0.003, +0.019], wins in 92% of resamples.

## s0-T004 Experiment: N1:ft, screen

Why: Opening: learned family, simplest candidate (based on s0-D001)

Macro AUROC 0.770 [0.725, 0.812], folds 0.769 ± 0.061; balanced accuracy 0.423 [0.351, 0.497]; 6 s.
- vs row 5: AUROC gain -0.029 [-0.053, -0.007], wins in 1% of resamples.
- vs best so far (s0-T003): AUROC gain -0.036 [-0.061, -0.011], wins in 0% of resamples.

## s0-T005 Experiment: M1:ft, screen

Why: Opening: mil family, simplest candidate (based on s0-D001)

Macro AUROC 0.854 [0.818, 0.885], folds 0.855 ± 0.019; balanced accuracy 0.546 [0.472, 0.620]; 141 s.
- vs row 5: AUROC gain +0.054 [+0.022, +0.084], wins in 100% of resamples.
- vs best so far (s0-T003): AUROC gain +0.047 [+0.018, +0.077], wins in 100% of resamples.

## s0-T006 Experiment: X1:raw, screen

Why: Opening: mexp family, simplest candidate (based on s0-D001)

Macro AUROC 0.858 [0.825, 0.887], folds 0.861 ± 0.026; balanced accuracy 0.539 [0.461, 0.614]; 114 s.
- vs row 5: AUROC gain +0.059 [+0.029, +0.085], wins in 100% of resamples.
- vs best so far (s0-T005): AUROC gain +0.004 [-0.007, +0.016], wins in 75% of resamples.

## s0-D002 Decision: leader

Leading family: mexp (best s0-T006 (X1:raw screen, AUROC 0.858)); explore it next.

Based on: s0-T006

- early: 0.786
- late: 0.799
- learned: 0.770
- mexp: 0.858
- mil: 0.854
- pool: 0.807

Action: `{"family": "mexp"}`

## s0-T007 Experiment: X2:raw, screen

Why: Explore the leading family mexp: untried recipe X2 at the default encoding raw (based on s0-D002, s0-T006)

Macro AUROC 0.856 [0.824, 0.886], folds 0.855 ± 0.035; balanced accuracy 0.525 [0.448, 0.600]; 109 s.
- vs row 5: AUROC gain +0.057 [+0.026, +0.084], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.002 [-0.009, +0.005], wins in 29% of resamples.
- Chosen per fold: [{'weights': [0.3166, 0.4599]}, {'weights': [0.3278, 0.1877]}, {'weights': [0.298, 0.2397]}, {'weights': [0.389, 0.3919]}, {'weights': [0.306, 0.2342]}, {'weights': [0.3147, 0.3416]}, {'weights': [0.2998, 0.4841]}, {'weights': [0.3411, 0.3394]}]

## s0-T008 Experiment: X3:raw, screen

Why: Explore the leading family mexp: untried recipe X3 at the default encoding raw (based on s0-D002, s0-T006)

Macro AUROC 0.853 [0.821, 0.884], folds 0.855 ± 0.042; balanced accuracy 0.505 [0.431, 0.578]; 3 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.079], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.004 [-0.014, +0.005], wins in 19% of resamples.
- Chosen per fold: [{'weights': [0.2825, 0.0559, 0.4644]}, {'weights': [0.2752, 0.0731, 0.1866]}, {'weights': [0.2654, 0.049, 0.2423]}, {'weights': [0.2871, 0.1184, 0.3707]}, {'weights': [0.2742, 0.0532, 0.2468]}, {'weights': [0.2776, 0.0545, 0.3512]}, {'weights': [0.3382, 0.0, 0.442]}, {'weights': [0.2736, 0.0842, 0.3395]}]

## s0-T009 Experiment: X4:raw, screen

Why: Explore the leading family mexp: untried recipe X4 at the default encoding raw (based on s0-D002, s0-T006)

Macro AUROC 0.856 [0.824, 0.886], folds 0.857 ± 0.035; balanced accuracy 0.509 [0.431, 0.586]; 2 s.
- vs row 5: AUROC gain +0.057 [+0.029, +0.082], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.002 [-0.011, +0.008], wins in 35% of resamples.
- Chosen per fold: [{'weights': [0.2757, 0.0714, 0.45]}, {'weights': [0.2855, 0.0862, 0.1696]}, {'weights': [0.2651, 0.0627, 0.2161]}, {'weights': [0.2948, 0.1326, 0.4038]}, {'weights': [0.2808, 0.056, 0.2382]}, {'weights': [0.2778, 0.0657, 0.3588]}, {'weights': [0.3585, 0.0, 0.4065]}, {'weights': [0.2751, 0.1005, 0.3352]}]

## s0-T010 Experiment: X1:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X1, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.854 [0.820, 0.885], folds 0.852 ± 0.028; balanced accuracy 0.511 [0.436, 0.585]; 2 s.
- vs row 5: AUROC gain +0.055 [+0.024, +0.082], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.004 [-0.010, +0.002], wins in 9% of resamples.

## s0-T011 Experiment: X4:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X4, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.855 [0.822, 0.885], folds 0.857 ± 0.034; balanced accuracy 0.527 [0.455, 0.601]; 3 s.
- vs row 5: AUROC gain +0.056 [+0.027, +0.081], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.003 [-0.013, +0.007], wins in 28% of resamples.
- Chosen per fold: [{'weights': [0.2794, 0.0709, 0.1423]}, {'weights': [0.2875, 0.0865, 0.0]}, {'weights': [0.2667, 0.0651, 0.0685]}, {'weights': [0.3066, 0.1226, 0.0009]}, {'weights': [0.2829, 0.0563, 0.0578]}, {'weights': [0.2774, 0.0733, 0.0007]}, {'weights': [0.3646, 0.0, 0.073]}, {'weights': [0.2758, 0.1035, 0.0886]}]

## s0-T012 Experiment: X2:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X2, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.856 [0.824, 0.886], folds 0.860 ± 0.031; balanced accuracy 0.513 [0.438, 0.587]; 2 s.
- vs row 5: AUROC gain +0.057 [+0.027, +0.084], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.002 [-0.009, +0.006], wins in 35% of resamples.
- Chosen per fold: [{'weights': [0.3202, 0.113]}, {'weights': [0.3301, 0.0]}, {'weights': [0.3028, 0.0503]}, {'weights': [0.3909, 0.0]}, {'weights': [0.3093, 0.0179]}, {'weights': [0.3237, 0.0]}, {'weights': [0.3041, 0.0618]}, {'weights': [0.3461, 0.0568]}]

## s0-T013 Experiment: X3:ple, screen

Why: Explore the leading family mexp: untried age-and-sex encoding ple for X3, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.853 [0.820, 0.884], folds 0.854 ± 0.036; balanced accuracy 0.499 [0.425, 0.571]; 2 s.
- vs row 5: AUROC gain +0.054 [+0.025, +0.079], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.005 [-0.015, +0.005], wins in 17% of resamples.
- Chosen per fold: [{'weights': [0.2849, 0.0567, 0.1219]}, {'weights': [0.2764, 0.0741, 0.0]}, {'weights': [0.2667, 0.0524, 0.0522]}, {'weights': [0.2972, 0.1116, 0.0]}, {'weights': [0.2743, 0.0556, 0.0385]}, {'weights': [0.276, 0.0644, 0.0]}, {'weights': [0.3438, 0.0, 0.0422]}, {'weights': [0.2749, 0.0871, 0.0579]}]

## s0-D003 Decision: leader

Leading family: mil (best s0-T005 (M1:ft screen, AUROC 0.854)); explore it next.

Based on: s0-T005

- early: 0.786
- late: 0.799
- learned: 0.770
- mil: 0.854
- pool: 0.807

Action: `{"family": "mil"}`

## s0-T014 Experiment: M2:ft, screen

Why: Explore the leading family mil: untried recipe M2 at the default encoding ft (based on s0-D003, s0-T005)

Macro AUROC 0.855 [0.821, 0.886], folds 0.857 ± 0.032; balanced accuracy 0.534 [0.459, 0.608]; 37 s.
- vs row 5: AUROC gain +0.055 [+0.025, +0.083], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.003 [-0.012, +0.006], wins in 22% of resamples.

## s0-T015 Experiment: M3:ft, screen

Why: Explore the leading family mil: untried recipe M3 at the default encoding ft (based on s0-D003, s0-T014)

Macro AUROC 0.850 [0.816, 0.882], folds 0.852 ± 0.021; balanced accuracy 0.508 [0.439, 0.578]; 108 s.
- vs row 5: AUROC gain +0.051 [+0.021, +0.078], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.008 [-0.018, +0.003], wins in 6% of resamples.

## s0-T016 Experiment: M2:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M2, the family's best recipe without it (based on s0-D003, s0-T014)

Macro AUROC 0.856 [0.822, 0.887], folds 0.855 ± 0.031; balanced accuracy 0.512 [0.439, 0.587]; 104 s.
- vs row 5: AUROC gain +0.057 [+0.025, +0.085], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.002 [-0.011, +0.007], wins in 32% of resamples.

## s0-T017 Experiment: M1:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M1, the family's best recipe without it (based on s0-D003, s0-T016)

Macro AUROC 0.856 [0.821, 0.888], folds 0.856 ± 0.023; balanced accuracy 0.546 [0.469, 0.619]; 100 s.
- vs row 5: AUROC gain +0.057 [+0.024, +0.086], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.002 [-0.013, +0.009], wins in 38% of resamples.

## s0-D004 Decision: tune_start

Start tuning (the exploration budget is used): the best screened candidate of each recipe, top 3: s0-T017 (M1:ple screen, AUROC 0.856), s0-T016 (M2:ple screen, AUROC 0.856), s0-T015 (M3:ft screen, AUROC 0.850).

Based on: s0-T017, s0-T016, s0-T015

- s0-T015: 0.850
- s0-T016: 0.856
- s0-T017: 0.856

Action: `{"tune": ["M1:ple", "M2:ple", "M3:ft"]}`

## s0-T018 Experiment: M1:ple, tuned

Why: Tune M1:ple: C or weight decay and size by inner CV (based on s0-D004)

Status: **timed_out** after 1802 s.

## s0-T019 Experiment: M2:ple, tuned

Why: Tune M2:ple: C or weight decay and size by inner CV (based on s0-D004)

Status: **timed_out** after 1802 s.

## s0-T020 Experiment: M3:ft, tuned

Why: Tune M3:ft: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.849 [0.815, 0.881], folds 0.853 ± 0.020; balanced accuracy 0.522 [0.450, 0.594]; 613 s.
- vs row 5: AUROC gain +0.050 [+0.021, +0.078], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.008 [-0.018, +0.002], wins in 5% of resamples.
- Chosen per fold: [{'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 0.1}]

## s0-D005 Decision: final_selection

s0-T006 (X1:raw screen, AUROC 0.858) has the highest AUROC but does not clearly beat the simpler s0-T005 (M1:ft screen, AUROC 0.854): it wins in 75% of resamples (needs ≥ 90%) and its balanced accuracy differs by -0.007 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T011, s0-T012, s0-T013, s0-T014, s0-T015, s0-T016, s0-T017, s0-T020

- s0-T001: bacc_difference 0.104, candidate E1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.072, mode screen
- s0-T002: bacc_difference 0.124, candidate L1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.059, mode screen
- s0-T003: bacc_difference 0.116, candidate P1:raw, candidate_wins_share 1.000, clearly_beaten True, gain 0.051, mode screen
- s0-T004: bacc_difference 0.115, candidate N1:ft, candidate_wins_share 1.000, clearly_beaten True, gain 0.087, mode screen
- s0-T005: bacc_difference -0.007, candidate M1:ft, candidate_wins_share 0.750, clearly_beaten False, gain 0.004, mode screen

Action: `{"candidate": "s0-T006", "winner": "s0-T005", "winner_candidate": "M1:ft", "winner_mode": "screen"}`

## s0-A001 Matched baseline uni2h_m: M1:ft, screen

Why: Matched baseline uni2h_m: winner M1:ft (screen) on uni2h alone (based on s0-T005)

Macro AUROC 0.853 [0.819, 0.885], folds 0.849 ± 0.039; balanced accuracy 0.523 [0.450, 0.594]; 21 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.078], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.005 [-0.018, +0.008], wins in 21% of resamples.

## s0-A002 Matched baseline virchow2_m: M1:ft, screen

Why: Matched baseline virchow2_m: winner M1:ft (screen) on virchow2 alone (based on s0-T005)

Macro AUROC 0.841 [0.802, 0.875], folds 0.837 ± 0.028; balanced accuracy 0.535 [0.461, 0.607]; 47 s.
- vs row 5: AUROC gain +0.042 [+0.003, +0.076], wins in 98% of resamples.
- vs best so far (s0-T006): AUROC gain -0.017 [-0.033, -0.003], wins in 1% of resamples.

## s0-A003 Matched baseline diag_no_meta: M1:ft, screen

Why: Matched baseline diag_no_meta: winner M1:ft (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T005)

Macro AUROC 0.857 [0.822, 0.888], folds 0.857 ± 0.023; balanced accuracy 0.545 [0.472, 0.617]; 63 s.
- vs row 5: AUROC gain +0.058 [+0.025, +0.087], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.001 [-0.012, +0.010], wins in 44% of resamples.

## s0-A004 Matched baseline 5m: M1:ft, screen

Why: Matched row 5: winner M1:ft (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T005)

Macro AUROC 0.851 [0.817, 0.882], folds 0.847 ± 0.043; balanced accuracy 0.512 [0.439, 0.581]; 19 s.
- vs row 5: AUROC gain +0.052 [+0.025, +0.077], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.007 [-0.020, +0.006], wins in 13% of resamples.
