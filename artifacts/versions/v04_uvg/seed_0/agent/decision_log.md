# Decision log, seed 0 (dev)

Models: uni2h, virchow2, genbio_pathfm + age and sex. Code 759bf07236ba2dfa091ff4486becfd68a749e10d.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-D001 Decision: open

Open with one candidate per family at fixed values: E1:raw, L1:raw, P1:raw, N1:ft, M1:ft.

Based on: 


Action: `{"candidates": ["E1:raw", "L1:raw", "P1:raw", "N1:ft", "M1:ft"]}`

## s0-T001 Experiment: E1:raw, screen

Why: Opening: early family, simplest candidate (based on s0-D001)

Macro AUROC 0.790 [0.744, 0.831], folds 0.802 ± 0.052; balanced accuracy 0.448 [0.374, 0.524]; 11 s.
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

Macro AUROC 0.834 [0.794, 0.872], folds 0.824 ± 0.030; balanced accuracy 0.508 [0.433, 0.581]; 69 s.
- vs row 5: AUROC gain +0.035 [-0.001, +0.066], wins in 97% of resamples.
- vs best so far (s0-T003): AUROC gain +0.029 [-0.001, +0.057], wins in 96% of resamples.

## s0-D002 Decision: leader

Leading family: mil (best s0-T005 (M1:ft screen, AUROC 0.834)); explore it next.

Based on: s0-T005

- early: 0.790
- late: 0.801
- learned: 0.770
- mil: 0.834
- pool: 0.805

Action: `{"family": "mil"}`

## s0-T006 Experiment: M2:ft, screen

Why: Explore the leading family mil: untried recipe M2 at the default encoding ft (based on s0-D002, s0-T005)

Macro AUROC 0.851 [0.813, 0.886], folds 0.848 ± 0.033; balanced accuracy 0.502 [0.431, 0.575]; 176 s.
- vs row 5: AUROC gain +0.052 [+0.014, +0.084], wins in 100% of resamples.
- vs best so far (s0-T005): AUROC gain +0.017 [+0.005, +0.030], wins in 100% of resamples.

## s0-T007 Experiment: M3:ft, screen

Why: Explore the leading family mil: untried recipe M3 at the default encoding ft (based on s0-D002, s0-T006)

Macro AUROC 0.825 [0.786, 0.864], folds 0.817 ± 0.029; balanced accuracy 0.492 [0.418, 0.565]; 189 s.
- vs row 5: AUROC gain +0.026 [-0.009, +0.056], wins in 94% of resamples.
- vs best so far (s0-T006): AUROC gain -0.025 [-0.040, -0.012], wins in 0% of resamples.

## s0-T008 Experiment: M2:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M2, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.851 [0.813, 0.885], folds 0.847 ± 0.029; balanced accuracy 0.511 [0.438, 0.585]; 176 s.
- vs row 5: AUROC gain +0.051 [+0.014, +0.083], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.000 [-0.004, +0.004], wins in 42% of resamples.

## s0-T009 Experiment: M1:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M1, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.835 [0.797, 0.872], folds 0.825 ± 0.032; balanced accuracy 0.508 [0.433, 0.581]; 182 s.
- vs row 5: AUROC gain +0.036 [+0.002, +0.066], wins in 98% of resamples.
- vs best so far (s0-T006): AUROC gain -0.016 [-0.029, -0.004], wins in 1% of resamples.

## s0-T010 Experiment: M3:ple, screen

Why: Explore the leading family mil: untried age-and-sex encoding ple for M3, the family's best recipe without it (based on s0-D002, s0-T006)

Macro AUROC 0.826 [0.787, 0.863], folds 0.817 ± 0.029; balanced accuracy 0.491 [0.416, 0.562]; 177 s.
- vs row 5: AUROC gain +0.027 [-0.006, +0.055], wins in 95% of resamples.
- vs best so far (s0-T006): AUROC gain -0.025 [-0.040, -0.010], wins in 0% of resamples.

## s0-D003 Decision: leader

Leading family: pool (best s0-T003 (P1:raw screen, AUROC 0.805)); explore it next.

Based on: s0-T003

- early: 0.790
- late: 0.801
- learned: 0.770
- pool: 0.805

Action: `{"family": "pool"}`

## s0-T011 Experiment: P2:raw, screen

Why: Explore the leading family pool: untried recipe P2 at the default encoding raw (based on s0-D003, s0-T003)

Macro AUROC 0.801 [0.761, 0.839], folds 0.802 ± 0.048; balanced accuracy 0.418 [0.345, 0.493]; 26 s.
- vs row 5: AUROC gain +0.002 [-0.019, +0.021], wins in 59% of resamples.
- vs best so far (s0-T006): AUROC gain -0.049 [-0.079, -0.017], wins in 0% of resamples.
- Chosen per fold: [{'weights': [0.5511, 0.4094]}, {'weights': [0.5127, 0.1882]}, {'weights': [0.4823, 0.2816]}, {'weights': [0.5935, 0.323]}, {'weights': [0.5615, 0.1947]}, {'weights': [0.4876, 0.3783]}, {'weights': [0.5172, 0.4522]}, {'weights': [0.5665, 0.3083]}]

## s0-D004 Decision: drop

Drop pool: the best experiment s0-T006 (M2:ft screen, AUROC 0.851) beats the pool family's best s0-T003 (P1:raw screen, AUROC 0.805) in 100% of resamples (≥ 90%).

Based on: s0-T006, s0-T003

- best_wins_share: 0.996
- gain: 0.046

Action: `{"family": "pool"}`

## s0-D005 Decision: leader

Leading family: late (best s0-T002 (L1:raw screen, AUROC 0.801)); explore it next.

Based on: s0-T002

- early: 0.790
- late: 0.801
- learned: 0.770

Action: `{"family": "late"}`

## s0-T012 Experiment: L2:raw, screen

Why: Explore the leading family late: untried recipe L2 at the default encoding raw (based on s0-D005, s0-T002)

Macro AUROC 0.807 [0.767, 0.843], folds 0.811 ± 0.044; balanced accuracy 0.428 [0.354, 0.499]; 7 s.
- vs row 5: AUROC gain +0.007 [-0.011, +0.023], wins in 78% of resamples.
- vs best so far (s0-T006): AUROC gain -0.044 [-0.072, -0.015], wins in 0% of resamples.

## s0-D006 Decision: drop

Drop late: the best experiment s0-T006 (M2:ft screen, AUROC 0.851) beats the late family's best s0-T012 (L2:raw screen, AUROC 0.807) in 100% of resamples (≥ 90%).

Based on: s0-T006, s0-T012

- best_wins_share: 0.997
- gain: 0.044

Action: `{"family": "late"}`

## s0-D007 Decision: leader

Leading family: early (best s0-T001 (E1:raw screen, AUROC 0.790)); explore it next.

Based on: s0-T001

- early: 0.790
- learned: 0.770

Action: `{"family": "early"}`

## s0-T013 Experiment: E2:raw, screen

Why: Explore the leading family early: untried recipe E2 at the default encoding raw (based on s0-D007, s0-T001)

Macro AUROC 0.788 [0.744, 0.829], folds 0.797 ± 0.051; balanced accuracy 0.429 [0.356, 0.502]; 10 s.
- vs row 5: AUROC gain -0.011 [-0.036, +0.011], wins in 17% of resamples.
- vs best so far (s0-T006): AUROC gain -0.063 [-0.096, -0.028], wins in 0% of resamples.

## s0-D008 Decision: drop

Drop early: the best experiment s0-T006 (M2:ft screen, AUROC 0.851) beats the early family's best s0-T001 (E1:raw screen, AUROC 0.790) in 100% of resamples (≥ 90%).

Based on: s0-T006, s0-T001

- best_wins_share: 1.000
- gain: 0.061

Action: `{"family": "early"}`

## s0-D009 Decision: leader

Leading family: learned (best s0-T004 (N1:ft screen, AUROC 0.770)); explore it next.

Based on: s0-T004

- learned: 0.770

Action: `{"family": "learned"}`

## s0-T014 Experiment: N2:ft, screen

Why: Explore the leading family learned: untried recipe N2 at the default encoding ft (based on s0-D009, s0-T004)

Macro AUROC 0.778 [0.734, 0.818], folds 0.773 ± 0.059; balanced accuracy 0.400 [0.328, 0.472]; 8 s.
- vs row 5: AUROC gain -0.021 [-0.051, +0.006], wins in 7% of resamples.
- vs best so far (s0-T006): AUROC gain -0.073 [-0.106, -0.039], wins in 0% of resamples.

## s0-D010 Decision: drop

Drop learned: the best experiment s0-T006 (M2:ft screen, AUROC 0.851) beats the learned family's best s0-T014 (N2:ft screen, AUROC 0.778) in 100% of resamples (≥ 90%).

Based on: s0-T006, s0-T014

- best_wins_share: 1.000
- gain: 0.073

Action: `{"family": "learned"}`

## s0-D011 Decision: tune_start

Start tuning (no untried candidate is left): the best screened candidate of each recipe, top 3: s0-T006 (M2:ft screen, AUROC 0.851), s0-T009 (M1:ple screen, AUROC 0.835), s0-T010 (M3:ple screen, AUROC 0.826).

Based on: s0-T006, s0-T009, s0-T010

- s0-T006: 0.851
- s0-T009: 0.835
- s0-T010: 0.826

Action: `{"tune": ["M2:ft", "M1:ple", "M3:ple"]}`

## s0-T015 Experiment: M2:ft, tuned

Why: Tune M2:ft: C or weight decay and size by inner CV (based on s0-D011)

Status: **timed_out** after 1802 s.

## s0-T016 Experiment: M1:ple, tuned

Why: Tune M1:ple: C or weight decay and size by inner CV (based on s0-D011)

Status: **timed_out** after 1802 s.

## s0-T017 Experiment: M3:ple, tuned

Why: Tune M3:ple: C or weight decay and size by inner CV (based on s0-D011)

Status: **timed_out** after 1802 s.

## s0-D012 Decision: final_selection

s0-T006 (M2:ft screen, AUROC 0.851) has the highest AUROC and clearly beats every simpler experiment of the families in play.

Based on: s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010

- s0-T005: bacc_difference -0.006, candidate M1:ft, candidate_wins_share 0.999, clearly_beaten True, gain 0.017, mode screen
- s0-T009: bacc_difference -0.006, candidate M1:ple, candidate_wins_share 0.994, clearly_beaten True, gain 0.016, mode screen

Action: `{"candidate": "s0-T006", "winner": "s0-T006", "winner_candidate": "M2:ft", "winner_mode": "screen"}`

## s0-A001 Matched baseline uni2h_m: M2:ft, screen

Why: Matched baseline uni2h_m: winner M2:ft (screen) on uni2h alone (based on s0-T006)

Macro AUROC 0.853 [0.819, 0.885], folds 0.849 ± 0.039; balanced accuracy 0.523 [0.450, 0.594]; 17 s.
- vs row 5: AUROC gain +0.054 [+0.026, +0.078], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain +0.002 [-0.017, +0.025], wins in 56% of resamples.

## s0-A002 Matched baseline virchow2_m: M2:ft, screen

Why: Matched baseline virchow2_m: winner M2:ft (screen) on virchow2 alone (based on s0-T006)

Macro AUROC 0.841 [0.802, 0.875], folds 0.837 ± 0.028; balanced accuracy 0.535 [0.461, 0.607]; 47 s.
- vs row 5: AUROC gain +0.042 [+0.003, +0.076], wins in 98% of resamples.
- vs best so far (s0-T006): AUROC gain -0.010 [-0.027, +0.007], wins in 14% of resamples.

## s0-A003 Matched baseline genbio_pathfm_m: M2:ft, screen

Why: Matched baseline genbio_pathfm_m: winner M2:ft (screen) on genbio_pathfm alone (based on s0-T006)

Macro AUROC 0.837 [0.800, 0.873], folds 0.841 ± 0.027; balanced accuracy 0.475 [0.401, 0.548]; 30 s.
- vs row 5: AUROC gain +0.038 [+0.004, +0.070], wins in 99% of resamples.
- vs best so far (s0-T006): AUROC gain -0.014 [-0.028, +0.001], wins in 4% of resamples.

## s0-A004 Matched baseline diag_no_meta: M2:ft, screen

Why: Matched baseline diag_no_meta: winner M2:ft (screen) on all image models without age and sex (metadata diagnostic) (based on s0-T006)

Macro AUROC 0.853 [0.816, 0.888], folds 0.850 ± 0.032; balanced accuracy 0.518 [0.444, 0.592]; 177 s.
- vs row 5: AUROC gain +0.054 [+0.017, +0.086], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain +0.003 [-0.001, +0.006], wins in 93% of resamples.

## s0-A005 Matched baseline 5m: M2:ft, screen

Why: Matched row 5: winner M2:ft (screen) on the best matched single model (uni2h) plus age and sex (based on s0-T006)

Macro AUROC 0.851 [0.817, 0.882], folds 0.847 ± 0.043; balanced accuracy 0.512 [0.439, 0.581]; 21 s.
- vs row 5: AUROC gain +0.052 [+0.025, +0.077], wins in 100% of resamples.
- vs best so far (s0-T006): AUROC gain -0.000 [-0.019, +0.024], wins in 49% of resamples.
