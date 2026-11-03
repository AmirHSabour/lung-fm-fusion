# Decision log, seed 0 (dev)

Models: uni2h, virchow2 + age and sex. Code 76b89b970be8853ff5842967ec3e7ca174851be8.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with one candidate per family at fixed values: E1:raw, L1:raw, P1:raw, N1:ft, M1:ft.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft", "M1:ft"]}`

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

Macro AUROC 0.770 [0.725, 0.812], folds 0.769 ± 0.061; balanced accuracy 0.423 [0.351, 0.497]; 7 s.
- vs row 5: AUROC gain -0.029 [-0.053, -0.007], wins in 1% of resamples.
- vs best so far (s0-T003): AUROC gain -0.036 [-0.061, -0.011], wins in 0% of resamples.

## s0-T005 Experiment: M1:ft, screen

Why: Opening: mil family, simplest candidate (based on s0-D001)

Macro AUROC 0.854 [0.818, 0.885], folds 0.855 ± 0.019; balanced accuracy 0.546 [0.472, 0.620]; 30 s.
- vs row 5: AUROC gain +0.054 [+0.022, +0.084], wins in 100% of resamples.
- vs best so far (s0-T003): AUROC gain +0.047 [+0.018, +0.077], wins in 100% of resamples.

## s0-D002 Decision: leader

Leading family: mil (best s0-T005 (M1:ft screen, AUROC 0.854)); explore it next.

Based on: s0-T005

- early: 0.786
- late: 0.799
- learned: 0.770
- mil: 0.854
- pool: 0.807

Action: `{"family": "mil"}`

## s0-T006 Experiment: M2:ft, screen

Why: Explore the leading family mil: untried recipe M2 at the default encoding ft (based on s0-D002, s0-T005)

Macro AUROC 0.855 [0.821, 0.886], folds 0.857 ± 0.032; balanced accuracy 0.534 [0.459, 0.608]; 26 s.
- vs row 5: AUROC gain +0.055 [+0.025, +0.083], wins in 100% of resamples.
- vs best so far (s0-T005): AUROC gain +0.001 [-0.010, +0.012], wins in 54% of resamples.

## s0-T007 Experiment: M3:ft, screen

Why: Explore the leading family mil: untried recipe M3 at the default encoding ft (based on s0-D002, s0-T006)

Macro AUROC 0.850 [0.816, 0.882], folds 0.852 ± 0.021; balanced accuracy 0.508 [0.439, 0.578]; 28 s.
- vs row 5: AUROC gain +0.051 [+0.021, +0.078], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.004 [-0.014, +0.005], wins in 19% of resamples.

## s0-T008 Experiment: M2:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M2, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.856 [0.822, 0.887], folds 0.855 ± 0.031; balanced accuracy 0.512 [0.439, 0.587]; 26 s.
- vs row 5: AUROC gain +0.057 [+0.025, +0.085], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain +0.001 [-0.003, +0.005], wins in 71% of resamples.

## s0-T009 Experiment: M1:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M1, the family's best recipe without it (based on s0-D002, s0-T008)

Macro AUROC 0.856 [0.821, 0.888], folds 0.856 ± 0.023; balanced accuracy 0.546 [0.469, 0.619]; 33 s.
- vs row 5: AUROC gain +0.057 [+0.024, +0.086], wins in 100% of resamples.
- vs best so far (s0-T008): AUROC gain +0.000 [-0.009, +0.009], wins in 54% of resamples.

## s0-T010 Experiment: M3:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M3, the family's best recipe without it (based on s0-D002, s0-T009)

Macro AUROC 0.852 [0.819, 0.882], folds 0.854 ± 0.019; balanced accuracy 0.531 [0.459, 0.605]; 28 s.
- vs row 5: AUROC gain +0.052 [+0.023, +0.080], wins in 100% of resamples.
- vs best so far (s0-T009): AUROC gain -0.004 [-0.011, +0.004], wins in 14% of resamples.

## s0-D003 Decision: leader

Leading family: pool (best s0-T003 (P1:raw screen, AUROC 0.807)); explore it next.

Based on: s0-T003

- early: 0.786
- late: 0.799
- learned: 0.770
- pool: 0.807

Action: `{"family": "pool"}`

## s0-T011 Experiment: P2:raw, screen

Why: Explore the leading family pool: untried recipe P2 at the default encoding raw (based on s0-D003, s0-T003)

Macro AUROC 0.803 [0.762, 0.839], folds 0.806 ± 0.052; balanced accuracy 0.420 [0.351, 0.491]; 12 s.
- vs row 5: AUROC gain +0.004 [-0.017, +0.022], wins in 64% of resamples.
- vs best so far (s0-T009): AUROC gain -0.053 [-0.082, -0.026], wins in 0% of resamples.
- Chosen per fold: [{'weights': [0.6039, 0.4136]}, {'weights': [0.5679, 0.1765]}, {'weights': [0.5171, 0.2785]}, {'weights': [0.6402, 0.3173]}, {'weights': [0.5941, 0.1978]}, {'weights': [0.5283, 0.3733]}, {'weights': [0.5617, 0.4411]}, {'weights': [0.5965, 0.2943]}]

## s0-D004 Decision: drop

Drop pool: the best experiment s0-T009 (M1:ple screen, AUROC 0.856) beats the pool family's best s0-T003 (P1:raw screen, AUROC 0.807) in 100% of resamples (≥ 90%).

Based on: s0-T009, s0-T003

- best_wins_share: 0.999
- gain: 0.049

Action: `{"family": "pool"}`

## s0-D005 Decision: leader

Leading family: late (best s0-T002 (L1:raw screen, AUROC 0.799)); explore it next.

Based on: s0-T002

- early: 0.786
- late: 0.799
- learned: 0.770

Action: `{"family": "late"}`

## s0-T012 Experiment: L2:raw, screen

Why: Explore the leading family late: untried recipe L2 at the default encoding raw (based on s0-D005, s0-T002)

Macro AUROC 0.808 [0.769, 0.843], folds 0.811 ± 0.049; balanced accuracy 0.431 [0.360, 0.503]; 4 s.
- vs row 5: AUROC gain +0.009 [-0.010, +0.025], wins in 84% of resamples.
- vs best so far (s0-T009): AUROC gain -0.048 [-0.074, -0.022], wins in 0% of resamples.

## s0-D006 Decision: drop

Drop late: the best experiment s0-T009 (M1:ple screen, AUROC 0.856) beats the late family's best s0-T012 (L2:raw screen, AUROC 0.808) in 100% of resamples (≥ 90%).

Based on: s0-T009, s0-T012

- best_wins_share: 1.000
- gain: 0.048

Action: `{"family": "late"}`

## s0-D007 Decision: leader

Leading family: early (best s0-T001 (E1:raw screen, AUROC 0.786)); explore it next.

Based on: s0-T001

- early: 0.786
- learned: 0.770

Action: `{"family": "early"}`

## s0-T013 Experiment: E2:raw, screen

Why: Explore the leading family early: untried recipe E2 at the default encoding raw (based on s0-D007, s0-T001)

Macro AUROC 0.784 [0.738, 0.827], folds 0.794 ± 0.060; balanced accuracy 0.433 [0.361, 0.506]; 5 s.
- vs row 5: AUROC gain -0.015 [-0.043, +0.008], wins in 10% of resamples.
- vs best so far (s0-T009): AUROC gain -0.072 [-0.107, -0.038], wins in 0% of resamples.

## s0-D008 Decision: drop

Drop early: the best experiment s0-T009 (M1:ple screen, AUROC 0.856) beats the early family's best s0-T001 (E1:raw screen, AUROC 0.786) in 100% of resamples (≥ 90%).

Based on: s0-T009, s0-T001

- best_wins_share: 1.000
- gain: 0.070

Action: `{"family": "early"}`

## s0-D009 Decision: leader

Leading family: learned (best s0-T004 (N1:ft screen, AUROC 0.770)); explore it next.

Based on: s0-T004

- learned: 0.770

Action: `{"family": "learned"}`

## s0-T014 Experiment: N2:ft, screen

Why: Explore the leading family learned: untried recipe N2 at the default encoding ft (based on s0-D009, s0-T004)

Macro AUROC 0.776 [0.730, 0.817], folds 0.775 ± 0.053; balanced accuracy 0.409 [0.339, 0.478]; 7 s.
- vs row 5: AUROC gain -0.023 [-0.057, +0.007], wins in 6% of resamples.
- vs best so far (s0-T009): AUROC gain -0.080 [-0.109, -0.052], wins in 0% of resamples.

## s0-D010 Decision: drop

Drop learned: the best experiment s0-T009 (M1:ple screen, AUROC 0.856) beats the learned family's best s0-T014 (N2:ft screen, AUROC 0.776) in 100% of resamples (≥ 90%).

Based on: s0-T009, s0-T014

- best_wins_share: 1.000
- gain: 0.080

Action: `{"family": "learned"}`

## s0-D011 Decision: tune_start

Start tuning (no untried candidate is left): the best screened candidate of each recipe, top 3: s0-T009 (M1:ple screen, AUROC 0.856), s0-T008 (M2:ple screen, AUROC 0.856), s0-T010 (M3:ple screen, AUROC 0.852).

Based on: s0-T009, s0-T008, s0-T010

- s0-T008: 0.856
- s0-T009: 0.856
- s0-T010: 0.852

Action: `{"tune": ["M1:ple", "M2:ple", "M3:ple"]}`

## s0-T015 Experiment: M1:ple, tuned

Why: Tune M1:ple: C or weight decay and size by inner CV (based on s0-D011)

Macro AUROC 0.855 [0.820, 0.887], folds 0.855 ± 0.025; balanced accuracy 0.546 [0.469, 0.619]; 547 s.
- vs row 5: AUROC gain +0.055 [+0.023, +0.085], wins in 100% of resamples.
- vs best so far (s0-T009): AUROC gain -0.001 [-0.004, +0.001], wins in 15% of resamples.
- Chosen per fold: [{'weight_decay': 1.0}, {'weight_decay': 0.0001}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 0.0001}]

## s0-T016 Experiment: M2:ple, tuned

Why: Tune M2:ple: C or weight decay and size by inner CV (based on s0-D011)

Macro AUROC 0.858 [0.825, 0.889], folds 0.859 ± 0.030; balanced accuracy 0.512 [0.439, 0.587]; 487 s.
- vs row 5: AUROC gain +0.059 [+0.027, +0.088], wins in 100% of resamples.
- vs best so far (s0-T009): AUROC gain +0.002 [-0.007, +0.011], wins in 64% of resamples.
- Chosen per fold: [{'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}]

## s0-T017 Experiment: M3:ple, tuned

Why: Tune M3:ple: C or weight decay and size by inner CV (based on s0-D011)

Macro AUROC 0.852 [0.818, 0.882], folds 0.855 ± 0.022; balanced accuracy 0.525 [0.450, 0.600]; 526 s.
- vs row 5: AUROC gain +0.052 [+0.023, +0.080], wins in 100% of resamples.
- vs best so far (s0-T016): AUROC gain -0.006 [-0.015, +0.002], wins in 8% of resamples.
- Chosen per fold: [{'weight_decay': 0.0001}, {'weight_decay': 0.1}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}, {'weight_decay': 1.0}]

## s0-D012 Decision: final_selection

s0-T016 (M2:ple tuned, AUROC 0.858) has the highest AUROC but does not clearly beat the simpler s0-T005 (M1:ft screen, AUROC 0.854): it wins in 80% of resamples (needs ≥ 90%) and its balanced accuracy differs by -0.034 (may be at most 0.02 lower); the simpler one is chosen.

Based on: s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T015, s0-T016, s0-T017

- s0-T005: bacc_difference -0.034, candidate M1:ft, candidate_wins_share 0.796, clearly_beaten False, gain 0.004, mode screen

Action: `{"candidate": "s0-T016", "winner": "s0-T005", "winner_candidate": "M1:ft", "winner_mode": "screen"}`

## s0-A001 Matched baseline uni2h_m: M1:ft, screen

Why: Matched baseline uni2h_m: winner M1:ft (screen) on uni2h alone (based on s0-T005)

Macro AUROC 0.853 [0.819, 0.885], folds 0.849 ± 0.039; balanced accuracy 0.523 [0.450, 0.594]; 14 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.078], wins in 100% of resamples.
- vs best so far (s0-T016): AUROC gain -0.005 [-0.018, +0.009], wins in 22% of resamples.

## s0-A002 Matched baseline virchow2_m: M1:ft, screen

Why: Matched baseline virchow2_m: winner M1:ft (screen) on virchow2 alone (based on s0-T005)

Macro AUROC 0.841 [0.802, 0.875], folds 0.837 ± 0.028; balanced accuracy 0.535 [0.461, 0.607]; 19 s.
- vs row 5: AUROC gain +0.042 [+0.003, +0.076], wins in 98% of resamples.
- vs best so far (s0-T016): AUROC gain -0.017 [-0.034, -0.003], wins in 1% of resamples.

## s0-A003 Matched baseline diag_no_meta: M1:ft, screen

Why: Matched baseline diag_no_meta: winner M1:ft (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T005)

Macro AUROC 0.857 [0.822, 0.888], folds 0.857 ± 0.023; balanced accuracy 0.545 [0.472, 0.617]; 28 s.
- vs row 5: AUROC gain +0.058 [+0.025, +0.087], wins in 100% of resamples.
- vs best so far (s0-T016): AUROC gain -0.001 [-0.011, +0.008], wins in 41% of resamples.

## s0-A004 Matched baseline 5m: M1:ft, screen

Why: Matched row 5: winner M1:ft (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T005)

Macro AUROC 0.851 [0.817, 0.882], folds 0.847 ± 0.043; balanced accuracy 0.512 [0.439, 0.581]; 14 s.
- vs row 5: AUROC gain +0.052 [+0.025, +0.077], wins in 100% of resamples.
- vs best so far (s0-T016): AUROC gain -0.007 [-0.021, +0.007], wins in 15% of resamples.
