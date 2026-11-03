# Decision log, seed 0 (dry run)

Models: uni2h, virchow2 + age and sex. Code ce3a4131546f787e6283bd0464e63bc80b28150f-dirty.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-T001 Experiment: E1, phase 1

Why: Phase 1 screening, recipe 1 of 11, fixed values, fixed order

Macro AUROC 0.786 [0.740, 0.828], folds 0.796 ± 0.056; balanced accuracy 0.435 [0.362, 0.507]; 5 s.
- vs row 5: AUROC gain -0.013 [-0.039, +0.009], wins in 12% of resamples.

## s0-T002 Experiment: E2, phase 1

Why: Phase 1 screening, recipe 2 of 11, fixed values, fixed order

Macro AUROC 0.784 [0.738, 0.827], folds 0.794 ± 0.060; balanced accuracy 0.433 [0.361, 0.506]; 5 s.
- vs row 5: AUROC gain -0.015 [-0.043, +0.008], wins in 10% of resamples.
- vs best so far (s0-T001): AUROC gain -0.002 [-0.007, +0.003], wins in 24% of resamples.

## s0-T003 Experiment: E3, phase 1, size 32

Why: Phase 1 screening, recipe 3 of 11, fixed values, fixed order

Macro AUROC 0.737 [0.690, 0.779], folds 0.751 ± 0.048; balanced accuracy 0.385 [0.313, 0.456]; 2 s.
- vs row 5: AUROC gain -0.062 [-0.102, -0.027], wins in 0% of resamples.
- vs best so far (s0-T001): AUROC gain -0.049 [-0.076, -0.021], wins in 0% of resamples.

## s0-T004 Experiment: E4, phase 1, size 32

Why: Phase 1 screening, recipe 4 of 11, fixed values, fixed order

Macro AUROC 0.739 [0.693, 0.781], folds 0.754 ± 0.049; balanced accuracy 0.390 [0.318, 0.462]; 3 s.
- vs row 5: AUROC gain -0.060 [-0.100, -0.025], wins in 0% of resamples.
- vs best so far (s0-T001): AUROC gain -0.047 [-0.073, -0.021], wins in 0% of resamples.

## s0-T005 Experiment: L1, phase 1

Why: Phase 1 screening, recipe 5 of 11, fixed values, fixed order

Macro AUROC 0.799 [0.757, 0.838], folds 0.808 ± 0.057; balanced accuracy 0.414 [0.345, 0.485]; 4 s.
- vs row 5: AUROC gain -0.000 [-0.023, +0.020], wins in 49% of resamples.
- vs best so far (s0-T001): AUROC gain +0.013 [-0.001, +0.028], wins in 97% of resamples.

## s0-T006 Experiment: L2, phase 1

Why: Phase 1 screening, recipe 6 of 11, fixed values, fixed order

Macro AUROC 0.808 [0.769, 0.843], folds 0.811 ± 0.049; balanced accuracy 0.431 [0.360, 0.503]; 4 s.
- vs row 5: AUROC gain +0.009 [-0.010, +0.025], wins in 84% of resamples.
- vs best so far (s0-T005): AUROC gain +0.009 [-0.000, +0.020], wins in 97% of resamples.

## s0-T007 Experiment: L3, phase 1

Why: Phase 1 screening, recipe 7 of 11, fixed values, fixed order

Macro AUROC 0.714 [0.667, 0.763], folds 0.716 ± 0.070; balanced accuracy 0.449 [0.379, 0.520]; 12 s.
- vs row 5: AUROC gain -0.085 [-0.123, -0.046], wins in 0% of resamples.
- vs best so far (s0-T006): AUROC gain -0.094 [-0.129, -0.060], wins in 0% of resamples.

## s0-T008 Experiment: L4, phase 1

Why: Phase 1 screening, recipe 8 of 11, fixed values, fixed order

Macro AUROC 0.806 [0.768, 0.840], folds 0.810 ± 0.030; balanced accuracy 0.433 [0.357, 0.510]; 12 s.
- vs row 5: AUROC gain +0.007 [-0.031, +0.042], wins in 65% of resamples.
- vs best so far (s0-T006): AUROC gain -0.002 [-0.033, +0.028], wins in 45% of resamples.

## s0-T009 Experiment: N1, phase 1, size 32

Why: Phase 1 screening, recipe 9 of 11, fixed values, fixed order

Macro AUROC 0.773 [0.728, 0.815], folds 0.771 ± 0.057; balanced accuracy 0.424 [0.348, 0.498]; 12 s.
- vs row 5: AUROC gain -0.026 [-0.061, +0.002], wins in 4% of resamples.
- vs best so far (s0-T006): AUROC gain -0.035 [-0.065, -0.008], wins in 0% of resamples.

## s0-T010 Experiment: N2, phase 1, size 32

Why: Phase 1 screening, recipe 10 of 11, fixed values, fixed order

Macro AUROC 0.783 [0.740, 0.821], folds 0.794 ± 0.040; balanced accuracy 0.446 [0.370, 0.524]; 7 s.
- vs row 5: AUROC gain -0.016 [-0.041, +0.006], wins in 8% of resamples.
- vs best so far (s0-T006): AUROC gain -0.025 [-0.048, -0.004], wins in 1% of resamples.

## s0-T011 Experiment: N3, phase 1, size 32

Why: Phase 1 screening, recipe 11 of 11, fixed values, fixed order

Macro AUROC 0.788 [0.745, 0.829], folds 0.795 ± 0.083; balanced accuracy 0.450 [0.371, 0.530]; 6 s.
- vs row 5: AUROC gain -0.011 [-0.043, +0.018], wins in 21% of resamples.
- vs best so far (s0-T006): AUROC gain -0.020 [-0.048, +0.006], wins in 7% of resamples.

## s0-D001 Decision: shortlist

Best in Phase 1: s0-T006 (L2, AUROC 0.808). A recipe stays in if the best does not win in at least 90% of paired resamples; 1 tied, keeping at most 3: s0-T006 (L2, AUROC 0.808), s0-T008 (L4, AUROC 0.806).

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T011

- s0-T001: best_wins_share 1.000, gain 0.022, recipe E1
- s0-T002: best_wins_share 1.000, gain 0.024, recipe E2
- s0-T003: best_wins_share 1.000, gain 0.071, recipe E3
- s0-T004: best_wins_share 1.000, gain 0.069, recipe E4
- s0-T005: best_wins_share 0.969, gain 0.009, recipe L1
- s0-T007: best_wins_share 1.000, gain 0.094, recipe L3
- s0-T008: best_wins_share 0.548, gain 0.002, recipe L4
- s0-T009: best_wins_share 0.995, gain 0.035, recipe N1
- s0-T010: best_wins_share 0.989, gain 0.025, recipe N2
- s0-T011: best_wins_share 0.932, gain 0.020, recipe N3

Action: `{"phase1_ids": ["s0-T006", "s0-T008"], "shortlist": ["L2", "L4"]}`

## s0-T012 Experiment: L2, phase 2

Why: Phase 2: tune L2 (C by inner CV); it has no size (based on s0-D001)

Macro AUROC 0.797 [0.754, 0.836], folds 0.807 ± 0.044; balanced accuracy 0.447 [0.374, 0.517]; 324 s.
- vs row 5: AUROC gain -0.002 [-0.025, +0.016], wins in 43% of resamples.
- vs best so far (s0-T006): AUROC gain -0.011 [-0.028, +0.004], wins in 8% of resamples.
- Chosen per fold: [{'M': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T013 Experiment: L4, phase 2

Why: Phase 2: tune L4 (C by inner CV); it has no size (based on s0-D001)

Macro AUROC 0.816 [0.784, 0.848], folds 0.816 ± 0.044; balanced accuracy 0.381 [0.312, 0.452]; 359 s.
- vs row 5: AUROC gain +0.017 [-0.016, +0.050], wins in 83% of resamples.
- vs best so far (s0-T006): AUROC gain +0.008 [-0.023, +0.038], wins in 70% of resamples.
- Chosen per fold: [{'M': 0.1, 'stack': 0.01, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'stack': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'stack': 0.01, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'stack': 1.0, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-D002 Decision: final_selection

s0-T013 (L4, AUROC 0.816) has the highest AUROC but does not clearly beat the simpler s0-T012 (L2, AUROC 0.797); the simpler recipe is chosen.

Based on: s0-T012, s0-T013, s0-D001

- s0-T012: bacc_difference -0.066, candidate_wins_share 0.865, clearly_beaten False, gain 0.020, recipe L2

Action: `{"candidate": "s0-T013", "winner": "s0-T012", "winner_phase": 2, "winner_recipe": "L2", "winner_size": null}`

## s0-A001 Ablation (row 5b): L2, phase 2

Why: Row 5b: the selected recipe's handling of age and sex, applied to row 5's model (uni2h) alone (based on s0-T012)

Macro AUROC 0.798 [0.757, 0.835], folds 0.803 ± 0.043; balanced accuracy 0.449 [0.378, 0.524]; 310 s.
- vs row 5: AUROC gain -0.001 [-0.014, +0.011], wins in 44% of resamples.
- vs best so far (s0-T013): AUROC gain -0.018 [-0.051, +0.014], wins in 14% of resamples.
- Chosen per fold: [{'M': 0.1, 'uni2h': 0.001}, {'M': 1.0, 'uni2h': 0.01}, {'M': 1.0, 'uni2h': 0.01}, {'M': 1.0, 'uni2h': 0.001}, {'M': 1.0, 'uni2h': 0.01}, {'M': 0.01, 'uni2h': 0.01}, {'M': 0.001, 'uni2h': 0.01}, {'M': 1.0, 'uni2h': 0.01}]
