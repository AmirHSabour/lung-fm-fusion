# Decision log, seed 0 (dev)

Models: uni2h, virchow2 + age and sex. Code 84f28662e8cb25f24006ffbadcee3ac38632996b-dirty.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with one candidate per family at fixed values: E1:raw, L1:raw, N1:ft.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "N1:ft"]}`

## s0-T001 Experiment: E1:raw, screen

Why: Opening: early family, simplest candidate (based on s0-D001)

Macro AUROC 0.786 [0.740, 0.828], folds 0.796 ± 0.056; balanced accuracy 0.435 [0.362, 0.507]; 5 s.
- vs row 5: AUROC gain -0.013 [-0.039, +0.009], wins in 12% of resamples.

## s0-T002 Experiment: L1:raw, screen

Why: Opening: late family, simplest candidate (based on s0-D001)

Macro AUROC 0.799 [0.757, 0.838], folds 0.808 ± 0.057; balanced accuracy 0.414 [0.345, 0.485]; 4 s.
- vs row 5: AUROC gain -0.000 [-0.023, +0.020], wins in 49% of resamples.
- vs best so far (s0-T001): AUROC gain +0.013 [-0.001, +0.028], wins in 97% of resamples.

## s0-T003 Experiment: N1:ft, screen

Why: Opening: learned family, simplest candidate (based on s0-D001)

Macro AUROC 0.770 [0.725, 0.812], folds 0.769 ± 0.061; balanced accuracy 0.423 [0.351, 0.497]; 7 s.
- vs row 5: AUROC gain -0.029 [-0.053, -0.007], wins in 1% of resamples.
- vs best so far (s0-T002): AUROC gain -0.029 [-0.054, -0.004], wins in 1% of resamples.

## s0-D002 Decision: leader

Leading family: late (best s0-T002 (L1:raw screen, AUROC 0.799)); explore it next.

Based on: s0-T002

- early: 0.786
- late: 0.799
- learned: 0.770

Action: `{"family": "late"}`

## s0-T004 Experiment: L2:raw, screen

Why: Explore the leading family late: untried recipe L2 at the default encoding raw (based on s0-D002, s0-T002)

Macro AUROC 0.808 [0.769, 0.843], folds 0.811 ± 0.049; balanced accuracy 0.431 [0.360, 0.503]; 4 s.
- vs row 5: AUROC gain +0.009 [-0.010, +0.025], wins in 84% of resamples.
- vs best so far (s0-T002): AUROC gain +0.009 [-0.000, +0.020], wins in 97% of resamples.

## s0-T005 Experiment: L3:raw, screen

Why: Explore the leading family late: untried recipe L3 at the default encoding raw (based on s0-D002, s0-T004)

Macro AUROC 0.714 [0.667, 0.763], folds 0.716 ± 0.070; balanced accuracy 0.449 [0.379, 0.520]; 12 s.
- vs row 5: AUROC gain -0.085 [-0.123, -0.046], wins in 0% of resamples.
- vs best so far (s0-T004): AUROC gain -0.094 [-0.129, -0.060], wins in 0% of resamples.

## s0-T006 Experiment: L4:raw, screen

Why: Explore the leading family late: untried recipe L4 at the default encoding raw (based on s0-D002, s0-T004)

Macro AUROC 0.806 [0.768, 0.840], folds 0.810 ± 0.030; balanced accuracy 0.433 [0.357, 0.510]; 12 s.
- vs row 5: AUROC gain +0.007 [-0.031, +0.042], wins in 65% of resamples.
- vs best so far (s0-T004): AUROC gain -0.002 [-0.033, +0.028], wins in 45% of resamples.

## s0-T007 Experiment: L2:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L2, the family's best recipe without it (based on s0-D002, s0-T004)

Macro AUROC 0.805 [0.768, 0.842], folds 0.813 ± 0.047; balanced accuracy 0.394 [0.325, 0.464]; 4 s.
- vs row 5: AUROC gain +0.006 [-0.015, +0.025], wins in 74% of resamples.
- vs best so far (s0-T004): AUROC gain -0.003 [-0.010, +0.005], wins in 24% of resamples.

## s0-T008 Experiment: L4:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L4, the family's best recipe without it (based on s0-D002, s0-T004)

Macro AUROC 0.812 [0.775, 0.846], folds 0.816 ± 0.038; balanced accuracy 0.464 [0.382, 0.542]; 12 s.
- vs row 5: AUROC gain +0.013 [-0.024, +0.046], wins in 75% of resamples.
- vs best so far (s0-T004): AUROC gain +0.004 [-0.030, +0.034], wins in 58% of resamples.

## s0-T009 Experiment: L1:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L1, the family's best recipe without it (based on s0-D002, s0-T008)

Macro AUROC 0.789 [0.747, 0.830], folds 0.791 ± 0.050; balanced accuracy 0.424 [0.352, 0.494]; 4 s.
- vs row 5: AUROC gain -0.010 [-0.031, +0.010], wins in 17% of resamples.
- vs best so far (s0-T008): AUROC gain -0.023 [-0.055, +0.012], wins in 10% of resamples.

## s0-T010 Experiment: L3:ple, screen

Why: Explore the leading family late: untried age-and-sex encoding ple for L3, the family's best recipe without it (based on s0-D002, s0-T008)

Macro AUROC 0.712 [0.664, 0.760], folds 0.710 ± 0.070; balanced accuracy 0.452 [0.380, 0.523]; 13 s.
- vs row 5: AUROC gain -0.087 [-0.126, -0.048], wins in 0% of resamples.
- vs best so far (s0-T008): AUROC gain -0.099 [-0.143, -0.055], wins in 0% of resamples.

## s0-D003 Decision: leader

Leading family: early (best s0-T001 (E1:raw screen, AUROC 0.786)); explore it next.

Based on: s0-T001

- early: 0.786
- learned: 0.770

Action: `{"family": "early"}`

## s0-T011 Experiment: E2:raw, screen

Why: Explore the leading family early: untried recipe E2 at the default encoding raw (based on s0-D003, s0-T001)

Macro AUROC 0.784 [0.738, 0.827], folds 0.794 ± 0.060; balanced accuracy 0.433 [0.361, 0.506]; 5 s.
- vs row 5: AUROC gain -0.015 [-0.043, +0.008], wins in 10% of resamples.
- vs best so far (s0-T008): AUROC gain -0.028 [-0.064, +0.007], wins in 6% of resamples.

## s0-D004 Decision: drop

Drop early: the best experiment s0-T008 (L4:ple screen, AUROC 0.812) beats the early family's best s0-T001 (E1:raw screen, AUROC 0.786) in 92% of resamples (≥ 90%).

Based on: s0-T008, s0-T001

- best_wins_share: 0.921
- gain: 0.026

Action: `{"family": "early"}`

## s0-D005 Decision: leader

Leading family: learned (best s0-T003 (N1:ft screen, AUROC 0.770)); explore it next.

Based on: s0-T003

- learned: 0.770

Action: `{"family": "learned"}`

## s0-T012 Experiment: N2:ft, screen

Why: Explore the leading family learned: untried recipe N2 at the default encoding ft (based on s0-D005, s0-T003)

Macro AUROC 0.776 [0.730, 0.817], folds 0.775 ± 0.053; balanced accuracy 0.409 [0.339, 0.478]; 7 s.
- vs row 5: AUROC gain -0.023 [-0.057, +0.007], wins in 6% of resamples.
- vs best so far (s0-T008): AUROC gain -0.036 [-0.075, +0.007], wins in 4% of resamples.

## s0-D006 Decision: drop

Drop learned: the best experiment s0-T008 (L4:ple screen, AUROC 0.812) beats the learned family's best s0-T012 (N2:ft screen, AUROC 0.776) in 96% of resamples (≥ 90%).

Based on: s0-T008, s0-T012

- best_wins_share: 0.957
- gain: 0.036

Action: `{"family": "learned"}`

## s0-D007 Decision: tune_start

Start tuning (no untried candidate is left): the best screened candidate of each recipe, top 3: s0-T008 (L4:ple screen, AUROC 0.812), s0-T004 (L2:raw screen, AUROC 0.808), s0-T002 (L1:raw screen, AUROC 0.799).

Based on: s0-T008, s0-T004, s0-T002

- s0-T002: 0.799
- s0-T004: 0.808
- s0-T008: 0.812

Action: `{"tune": ["L4:ple", "L2:raw", "L1:raw"]}`

## s0-T013 Experiment: L4:ple, tuned

Why: Tune L4:ple: C or weight decay and size by inner CV (based on s0-D007)

Macro AUROC 0.755 [0.712, 0.794], folds 0.799 ± 0.066; balanced accuracy 0.407 [0.334, 0.479]; 360 s.
- vs row 5: AUROC gain -0.044 [-0.085, -0.004], wins in 2% of resamples.
- vs best so far (s0-T008): AUROC gain -0.057 [-0.089, -0.023], wins in 0% of resamples.
- Chosen per fold: [{'M': 0.1, 'stack': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 0.0001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 0.0001, 'stack': 0.001, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 0.01, 'stack': 0.1, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.0001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T014 Experiment: L2:raw, tuned

Why: Tune L2:raw: C or weight decay and size by inner CV (based on s0-D007)

Macro AUROC 0.797 [0.754, 0.836], folds 0.807 ± 0.044; balanced accuracy 0.447 [0.374, 0.517]; 318 s.
- vs row 5: AUROC gain -0.002 [-0.025, +0.016], wins in 43% of resamples.
- vs best so far (s0-T008): AUROC gain -0.015 [-0.052, +0.024], wins in 22% of resamples.
- Chosen per fold: [{'M': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T015 Experiment: L1:raw, tuned

Why: Tune L1:raw: C or weight decay and size by inner CV (based on s0-D007)

Macro AUROC 0.794 [0.752, 0.833], folds 0.804 ± 0.050; balanced accuracy 0.429 [0.359, 0.500]; 318 s.
- vs row 5: AUROC gain -0.005 [-0.024, +0.011], wins in 30% of resamples.
- vs best so far (s0-T008): AUROC gain -0.017 [-0.051, +0.018], wins in 17% of resamples.
- Chosen per fold: [{'M': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-D008 Decision: final_selection

s0-T008 (L4:ple screen, AUROC 0.812) has the highest AUROC but does not clearly beat the simpler s0-T002 (L1:raw screen, AUROC 0.799): it wins in 78% of resamples (needs ≥ 90%) and its balanced accuracy differs by +0.050 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T002, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T013, s0-T014, s0-T015

- s0-T002: bacc_difference 0.050, candidate L1:raw, candidate_wins_share 0.775, clearly_beaten False, gain 0.013, mode screen

Action: `{"candidate": "s0-T008", "winner": "s0-T002", "winner_candidate": "L1:raw", "winner_mode": "screen"}`

## s0-A001 Matched baseline 1m: L1:raw, screen

Why: Matched baseline 1m: winner L1:raw (screen) on age and sex alone (based on s0-T002)

Macro AUROC 0.611 [0.560, 0.659], folds 0.634 ± 0.072; balanced accuracy 0.266 [0.208, 0.329]; 2 s.
- vs row 5: AUROC gain -0.188 [-0.252, -0.131], wins in 0% of resamples.
- vs best so far (s0-T008): AUROC gain -0.201 [-0.251, -0.154], wins in 0% of resamples.

## s0-A002 Matched baseline uni2h_m: L1:raw, screen

Why: Matched baseline uni2h_m: winner L1:raw (screen) on uni2h alone (based on s0-T002)

Macro AUROC 0.805 [0.766, 0.842], folds 0.808 ± 0.033; balanced accuracy 0.457 [0.385, 0.528]; 3 s.
- vs row 5: AUROC gain +0.006 [-0.006, +0.017], wins in 85% of resamples.
- vs best so far (s0-T008): AUROC gain -0.007 [-0.038, +0.028], wins in 35% of resamples.

## s0-A003 Matched baseline virchow2_m: L1:raw, screen

Why: Matched baseline virchow2_m: winner L1:raw (screen) on virchow2 alone (based on s0-T002)

Macro AUROC 0.794 [0.754, 0.832], folds 0.804 ± 0.045; balanced accuracy 0.422 [0.351, 0.495]; 4 s.
- vs row 5: AUROC gain -0.005 [-0.028, +0.016], wins in 32% of resamples.
- vs best so far (s0-T008): AUROC gain -0.018 [-0.053, +0.020], wins in 17% of resamples.

## s0-A004 Matched baseline diag_no_meta: L1:raw, screen

Why: Matched baseline diag_no_meta: winner L1:raw (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T002)

Macro AUROC 0.806 [0.768, 0.842], folds 0.810 ± 0.046; balanced accuracy 0.408 [0.339, 0.477]; 10 s.
- vs row 5: AUROC gain +0.007 [-0.007, +0.020], wins in 82% of resamples.
- vs best so far (s0-T008): AUROC gain -0.006 [-0.037, +0.029], wins in 36% of resamples.

## s0-A005 Matched baseline 5m: L1:raw, screen

Why: Matched row 5: winner L1:raw (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T002)

Macro AUROC 0.794 [0.751, 0.833], folds 0.798 ± 0.061; balanced accuracy 0.462 [0.392, 0.536]; 3 s.
- vs row 5: AUROC gain -0.005 [-0.029, +0.018], wins in 35% of resamples.
- vs best so far (s0-T008): AUROC gain -0.017 [-0.048, +0.015], wins in 15% of resamples.
