# Decision log, seed 0 (dev)

Models: uni2h, virchow2 + age and sex. Code 3bdd23e884d3e7c0111b2fe24348d35b56f4d7b0.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with one candidate per family at fixed values: E1:raw, L1:raw, P1:raw, N1:ft.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft"]}`

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

Macro AUROC 0.807 [0.766, 0.844], folds 0.811 ± 0.060; balanced accuracy 0.423 [0.350, 0.497]; 5 s.
- vs row 5: AUROC gain +0.008 [-0.015, +0.027], wins in 76% of resamples.
- vs best so far (s0-T002): AUROC gain +0.008 [-0.003, +0.019], wins in 92% of resamples.

## s0-T004 Experiment: N1:ft, screen

Why: Opening: learned family, simplest candidate (based on s0-D001)

Macro AUROC 0.770 [0.725, 0.812], folds 0.769 ± 0.061; balanced accuracy 0.423 [0.351, 0.497]; 6 s.
- vs row 5: AUROC gain -0.029 [-0.053, -0.007], wins in 1% of resamples.
- vs best so far (s0-T003): AUROC gain -0.036 [-0.061, -0.011], wins in 0% of resamples.

## s0-D002 Decision: leader

Leading family: pool (best s0-T003 (P1:raw screen, AUROC 0.807)); explore it next.

Based on: s0-T003

- early: 0.786
- late: 0.799
- learned: 0.770
- pool: 0.807

Action: `{"family": "pool"}`

## s0-T005 Experiment: P2:raw, screen

Why: Explore the leading family pool: untried recipe P2 at the default encoding raw (based on s0-D002, s0-T003)

Macro AUROC 0.803 [0.762, 0.839], folds 0.806 ± 0.052; balanced accuracy 0.420 [0.351, 0.491]; 12 s.
- vs row 5: AUROC gain +0.004 [-0.017, +0.022], wins in 64% of resamples.
- vs best so far (s0-T003): AUROC gain -0.004 [-0.011, +0.003], wins in 14% of resamples.
- Chosen per fold: [{'weights': [0.6039, 0.4136]}, {'weights': [0.5679, 0.1765]}, {'weights': [0.5171, 0.2785]}, {'weights': [0.6402, 0.3173]}, {'weights': [0.5941, 0.1978]}, {'weights': [0.5283, 0.3733]}, {'weights': [0.5617, 0.4411]}, {'weights': [0.5965, 0.2943]}]

## s0-T006 Experiment: P3:raw, screen

Why: Explore the leading family pool: untried recipe P3 at the default encoding raw (based on s0-D002, s0-T003)

Macro AUROC 0.802 [0.762, 0.840], folds 0.807 ± 0.039; balanced accuracy 0.453 [0.382, 0.526]; 13 s.
- vs row 5: AUROC gain +0.003 [-0.015, +0.020], wins in 65% of resamples.
- vs best so far (s0-T003): AUROC gain -0.005 [-0.014, +0.006], wins in 20% of resamples.
- Chosen per fold: [{'weights': [0.6107, 0.0457, 0.4122]}, {'weights': [0.6083, 0.0246, 0.1447]}, {'weights': [0.5805, 0.0, 0.2631]}, {'weights': [0.4006, 0.2527, 0.3052]}, {'weights': [0.3844, 0.2224, 0.199]}, {'weights': [0.4985, 0.0715, 0.3661]}, {'weights': [0.5499, 0.0535, 0.4128]}, {'weights': [0.4664, 0.1588, 0.2896]}]

## s0-T007 Experiment: P4:raw, screen

Why: Explore the leading family pool: untried recipe P4 at the default encoding raw (based on s0-D002, s0-T003)

Macro AUROC 0.803 [0.764, 0.840], folds 0.809 ± 0.036; balanced accuracy 0.484 [0.409, 0.563]; 12 s.
- vs row 5: AUROC gain +0.004 [-0.014, +0.020], wins in 70% of resamples.
- vs best so far (s0-T003): AUROC gain -0.004 [-0.014, +0.008], wins in 27% of resamples.
- Chosen per fold: [{'weights': [0.6094, 0.0563, 0.4181]}, {'weights': [0.6524, 0.0158, 0.1284]}, {'weights': [0.598, 0.0, 0.2586]}, {'weights': [0.3975, 0.2789, 0.3216]}, {'weights': [0.3875, 0.2355, 0.1958]}, {'weights': [0.4951, 0.0862, 0.3656]}, {'weights': [0.5609, 0.0681, 0.4048]}, {'weights': [0.4897, 0.1611, 0.2866]}]

## s0-T008 Experiment: P1:ple, screen

Why: Explore the leading family pool: untried age-and-sex encoding ple for P1, the family's best recipe without it (based on s0-D002, s0-T003)

Macro AUROC 0.793 [0.750, 0.833], folds 0.794 ± 0.054; balanced accuracy 0.370 [0.301, 0.443]; 4 s.
- vs row 5: AUROC gain -0.006 [-0.035, +0.017], wins in 31% of resamples.
- vs best so far (s0-T003): AUROC gain -0.014 [-0.026, -0.002], wins in 1% of resamples.

## s0-T009 Experiment: P4:ple, screen

Why: Explore the leading family pool: untried age-and-sex encoding ple for P4, the family's best recipe without it (based on s0-D002, s0-T003)

Macro AUROC 0.803 [0.764, 0.840], folds 0.810 ± 0.037; balanced accuracy 0.479 [0.407, 0.556]; 12 s.
- vs row 5: AUROC gain +0.003 [-0.011, +0.017], wins in 68% of resamples.
- vs best so far (s0-T003): AUROC gain -0.004 [-0.016, +0.010], wins in 28% of resamples.
- Chosen per fold: [{'weights': [0.6126, 0.0635, 0.1493]}, {'weights': [0.6634, 0.0129, 0.0082]}, {'weights': [0.6084, 0.0, 0.0974]}, {'weights': [0.4332, 0.2633, 0.0]}, {'weights': [0.3967, 0.234, 0.0626]}, {'weights': [0.5073, 0.0988, 0.0495]}, {'weights': [0.5777, 0.0665, 0.0861]}, {'weights': [0.5017, 0.1637, 0.093]}]

## s0-T010 Experiment: P2:ple, screen

Why: Explore the leading family pool: untried age-and-sex encoding ple for P2, the family's best recipe without it (based on s0-D002, s0-T003)

Macro AUROC 0.802 [0.764, 0.838], folds 0.806 ± 0.046; balanced accuracy 0.417 [0.347, 0.486]; 12 s.
- vs row 5: AUROC gain +0.003 [-0.014, +0.019], wins in 62% of resamples.
- vs best so far (s0-T003): AUROC gain -0.005 [-0.016, +0.007], wins in 20% of resamples.
- Chosen per fold: [{'weights': [0.6168, 0.1195]}, {'weights': [0.5781, 0.0]}, {'weights': [0.5292, 0.0741]}, {'weights': [0.6575, 0.0]}, {'weights': [0.6028, 0.0403]}, {'weights': [0.5581, 0.0271]}, {'weights': [0.5763, 0.0689]}, {'weights': [0.6124, 0.0711]}]

## s0-T011 Experiment: P3:ple, screen

Why: Explore the leading family pool: untried age-and-sex encoding ple for P3, the family's best recipe without it (based on s0-D002, s0-T003)

Macro AUROC 0.803 [0.764, 0.840], folds 0.807 ± 0.035; balanced accuracy 0.449 [0.377, 0.521]; 12 s.
- vs row 5: AUROC gain +0.004 [-0.010, +0.017], wins in 69% of resamples.
- vs best so far (s0-T003): AUROC gain -0.004 [-0.017, +0.010], wins in 29% of resamples.
- Chosen per fold: [{'weights': [0.6159, 0.0523, 0.1243]}, {'weights': [0.6216, 0.0212, 0.0]}, {'weights': [0.5926, 0.0, 0.0801]}, {'weights': [0.4334, 0.2408, 0.0]}, {'weights': [0.3914, 0.2239, 0.0463]}, {'weights': [0.5137, 0.0844, 0.0289]}, {'weights': [0.5696, 0.0503, 0.0591]}, {'weights': [0.4796, 0.162, 0.0715]}]

## s0-D003 Decision: leader

Leading family: late (best s0-T002 (L1:raw screen, AUROC 0.799)); explore it next.

Based on: s0-T002

- early: 0.786
- late: 0.799
- learned: 0.770

Action: `{"family": "late"}`

## s0-T012 Experiment: L2:raw, screen

Why: Explore the leading family late: untried recipe L2 at the default encoding raw (based on s0-D003, s0-T002)

Macro AUROC 0.808 [0.769, 0.843], folds 0.811 ± 0.049; balanced accuracy 0.431 [0.360, 0.503]; 4 s.
- vs row 5: AUROC gain +0.009 [-0.010, +0.025], wins in 84% of resamples.
- vs best so far (s0-T003): AUROC gain +0.001 [-0.006, +0.009], wins in 62% of resamples.

## s0-T013 Experiment: L3:raw, screen

Why: Explore the leading family late: untried recipe L3 at the default encoding raw (based on s0-D003, s0-T012)

Macro AUROC 0.714 [0.667, 0.763], folds 0.716 ± 0.070; balanced accuracy 0.449 [0.379, 0.520]; 12 s.
- vs row 5: AUROC gain -0.085 [-0.123, -0.046], wins in 0% of resamples.
- vs best so far (s0-T012): AUROC gain -0.094 [-0.129, -0.060], wins in 0% of resamples.

## s0-T014 Experiment: L4:raw, screen

Why: Explore the leading family late: untried recipe L4 at the default encoding raw (based on s0-D003, s0-T012)

Macro AUROC 0.806 [0.768, 0.840], folds 0.810 ± 0.030; balanced accuracy 0.433 [0.357, 0.510]; 13 s.
- vs row 5: AUROC gain +0.007 [-0.031, +0.042], wins in 65% of resamples.
- vs best so far (s0-T012): AUROC gain -0.002 [-0.033, +0.028], wins in 45% of resamples.

## s0-T015 Experiment: L2:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L2, the family's best recipe without it (based on s0-D003, s0-T012)

Macro AUROC 0.805 [0.768, 0.842], folds 0.813 ± 0.047; balanced accuracy 0.394 [0.325, 0.464]; 4 s.
- vs row 5: AUROC gain +0.006 [-0.015, +0.025], wins in 74% of resamples.
- vs best so far (s0-T012): AUROC gain -0.003 [-0.010, +0.005], wins in 24% of resamples.

## s0-T016 Experiment: L4:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L4, the family's best recipe without it (based on s0-D003, s0-T012)

Macro AUROC 0.812 [0.775, 0.846], folds 0.816 ± 0.038; balanced accuracy 0.464 [0.382, 0.542]; 12 s.
- vs row 5: AUROC gain +0.013 [-0.024, +0.046], wins in 75% of resamples.
- vs best so far (s0-T012): AUROC gain +0.004 [-0.030, +0.034], wins in 58% of resamples.

## s0-T017 Experiment: L1:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L1, the family's best recipe without it (based on s0-D003, s0-T016)

Macro AUROC 0.789 [0.747, 0.830], folds 0.791 ± 0.050; balanced accuracy 0.424 [0.352, 0.494]; 4 s.
- vs row 5: AUROC gain -0.010 [-0.031, +0.010], wins in 17% of resamples.
- vs best so far (s0-T016): AUROC gain -0.023 [-0.055, +0.012], wins in 10% of resamples.

## s0-D004 Decision: tune_start

Start tuning (the exploration budget is used): the best screened candidate of each recipe, top 3: s0-T016 (L4:ple screen, AUROC 0.812), s0-T012 (L2:raw screen, AUROC 0.808), s0-T003 (P1:raw screen, AUROC 0.807).

Based on: s0-T016, s0-T012, s0-T003

- s0-T003: 0.807
- s0-T012: 0.808
- s0-T016: 0.812

Action: `{"tune": ["L4:ple", "L2:raw", "P1:raw"]}`

## s0-T018 Experiment: L4:ple, tuned

Why: Tune L4:ple: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.755 [0.712, 0.794], folds 0.799 ± 0.066; balanced accuracy 0.407 [0.334, 0.479]; 359 s.
- vs row 5: AUROC gain -0.044 [-0.085, -0.004], wins in 2% of resamples.
- vs best so far (s0-T016): AUROC gain -0.057 [-0.089, -0.023], wins in 0% of resamples.
- Chosen per fold: [{'M': 0.1, 'stack': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 0.0001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 0.0001, 'stack': 0.001, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 0.01, 'stack': 0.1, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.0001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T019 Experiment: L2:raw, tuned

Why: Tune L2:raw: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.797 [0.754, 0.836], folds 0.807 ± 0.044; balanced accuracy 0.447 [0.374, 0.517]; 318 s.
- vs row 5: AUROC gain -0.002 [-0.025, +0.016], wins in 43% of resamples.
- vs best so far (s0-T016): AUROC gain -0.015 [-0.052, +0.024], wins in 22% of resamples.
- Chosen per fold: [{'M': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T020 Experiment: P1:raw, tuned

Why: Tune P1:raw: C or weight decay and size by inner CV (based on s0-D004)

Macro AUROC 0.800 [0.754, 0.840], folds 0.811 ± 0.053; balanced accuracy 0.440 [0.370, 0.511]; 317 s.
- vs row 5: AUROC gain +0.001 [-0.024, +0.020], wins in 54% of resamples.
- vs best so far (s0-T016): AUROC gain -0.012 [-0.050, +0.026], wins in 26% of resamples.
- Chosen per fold: [{'M': 0.1, 'uni2h': 0.001, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001, 'weights': [1.0, 1.0]}, {'M': 1.0, 'uni2h': 0.001, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 0.01, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 0.001, 'uni2h': 0.01, 'virchow2': 0.01, 'weights': [1.0, 1.0]}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 1.0, 'weights': [1.0, 1.0]}]

## s0-D005 Decision: final_selection

s0-T016 (L4:ple screen, AUROC 0.812) has the highest AUROC but does not clearly beat the simpler s0-T002 (L1:raw screen, AUROC 0.799): it wins in 78% of resamples (needs ≥ 90%) and its balanced accuracy differs by +0.050 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T011, s0-T012, s0-T013, s0-T014, s0-T015, s0-T016, s0-T017, s0-T018, s0-T019, s0-T020

- s0-T001: bacc_difference 0.029, candidate E1:raw, candidate_wins_share 0.921, clearly_beaten True, gain 0.026, mode screen
- s0-T002: bacc_difference 0.050, candidate L1:raw, candidate_wins_share 0.775, clearly_beaten False, gain 0.013, mode screen

Action: `{"candidate": "s0-T016", "winner": "s0-T002", "winner_candidate": "L1:raw", "winner_mode": "screen"}`

## s0-A001 Matched baseline 1m: L1:raw, screen

Why: Matched baseline 1m: winner L1:raw (screen) on age and sex alone (based on s0-T002)

Macro AUROC 0.611 [0.560, 0.659], folds 0.634 ± 0.072; balanced accuracy 0.266 [0.208, 0.329]; 2 s.
- vs row 5: AUROC gain -0.188 [-0.252, -0.131], wins in 0% of resamples.
- vs best so far (s0-T016): AUROC gain -0.201 [-0.251, -0.154], wins in 0% of resamples.

## s0-A002 Matched baseline uni2h_m: L1:raw, screen

Why: Matched baseline uni2h_m: winner L1:raw (screen) on uni2h alone (based on s0-T002)

Macro AUROC 0.805 [0.766, 0.842], folds 0.808 ± 0.033; balanced accuracy 0.457 [0.385, 0.528]; 3 s.
- vs row 5: AUROC gain +0.006 [-0.006, +0.017], wins in 85% of resamples.
- vs best so far (s0-T016): AUROC gain -0.007 [-0.038, +0.028], wins in 35% of resamples.

## s0-A003 Matched baseline virchow2_m: L1:raw, screen

Why: Matched baseline virchow2_m: winner L1:raw (screen) on virchow2 alone (based on s0-T002)

Macro AUROC 0.794 [0.754, 0.832], folds 0.804 ± 0.045; balanced accuracy 0.422 [0.351, 0.495]; 3 s.
- vs row 5: AUROC gain -0.005 [-0.028, +0.016], wins in 32% of resamples.
- vs best so far (s0-T016): AUROC gain -0.018 [-0.053, +0.020], wins in 17% of resamples.

## s0-A004 Matched baseline diag_no_meta: L1:raw, screen

Why: Matched baseline diag_no_meta: winner L1:raw (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T002)

Macro AUROC 0.806 [0.768, 0.842], folds 0.810 ± 0.046; balanced accuracy 0.408 [0.339, 0.477]; 4 s.
- vs row 5: AUROC gain +0.007 [-0.007, +0.020], wins in 82% of resamples.
- vs best so far (s0-T016): AUROC gain -0.006 [-0.037, +0.029], wins in 36% of resamples.

## s0-A005 Matched baseline 5m: L1:raw, screen

Why: Matched row 5: winner L1:raw (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T002)

Macro AUROC 0.794 [0.751, 0.833], folds 0.798 ± 0.061; balanced accuracy 0.462 [0.392, 0.536]; 3 s.
- vs row 5: AUROC gain -0.005 [-0.029, +0.018], wins in 35% of resamples.
- vs best so far (s0-T016): AUROC gain -0.017 [-0.048, +0.015], wins in 15% of resamples.
