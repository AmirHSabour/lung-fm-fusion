# Decision log, seed 0 (dev)

Models: uni2h, virchow2, genbio_pathfm + age and sex. Code 759bf07236ba2dfa091ff4486becfd68a749e10d.

Reference, row 5 (best single model + age and sex, plain concatenation): macro AUROC 0.799, balanced accuracy 0.456.

## s0-T001 Experiment: E1, phase 1

Why: Phase 1 screening, recipe 1 of 11, fixed values, fixed order

Macro AUROC 0.790 [0.744, 0.831], folds 0.802 ± 0.052; balanced accuracy 0.448 [0.374, 0.524]; 12 s.
- vs row 5: AUROC gain -0.010 [-0.035, +0.013], wins in 20% of resamples.

## s0-T002 Experiment: E2, phase 1

Why: Phase 1 screening, recipe 2 of 11, fixed values, fixed order

Macro AUROC 0.788 [0.744, 0.829], folds 0.797 ± 0.051; balanced accuracy 0.429 [0.356, 0.502]; 10 s.
- vs row 5: AUROC gain -0.011 [-0.036, +0.011], wins in 17% of resamples.
- vs best so far (s0-T001): AUROC gain -0.001 [-0.006, +0.003], wins in 29% of resamples.

## s0-T003 Experiment: E3, phase 1, size 32

Why: Phase 1 screening, recipe 3 of 11, fixed values, fixed order

Macro AUROC 0.746 [0.699, 0.788], folds 0.754 ± 0.043; balanced accuracy 0.385 [0.312, 0.460]; 3 s.
- vs row 5: AUROC gain -0.053 [-0.088, -0.022], wins in 0% of resamples.
- vs best so far (s0-T001): AUROC gain -0.043 [-0.067, -0.020], wins in 0% of resamples.

## s0-T004 Experiment: E4, phase 1, size 32

Why: Phase 1 screening, recipe 4 of 11, fixed values, fixed order

Macro AUROC 0.747 [0.701, 0.788], folds 0.759 ± 0.042; balanced accuracy 0.373 [0.303, 0.444]; 3 s.
- vs row 5: AUROC gain -0.052 [-0.088, -0.021], wins in 0% of resamples.
- vs best so far (s0-T001): AUROC gain -0.043 [-0.065, -0.019], wins in 0% of resamples.

## s0-T005 Experiment: L1, phase 1

Why: Phase 1 screening, recipe 5 of 11, fixed values, fixed order

Macro AUROC 0.801 [0.760, 0.839], folds 0.810 ± 0.051; balanced accuracy 0.425 [0.351, 0.496]; 7 s.
- vs row 5: AUROC gain +0.002 [-0.020, +0.021], wins in 57% of resamples.
- vs best so far (s0-T001): AUROC gain +0.012 [-0.002, +0.026], wins in 95% of resamples.

## s0-T006 Experiment: L2, phase 1

Why: Phase 1 screening, recipe 6 of 11, fixed values, fixed order

Macro AUROC 0.807 [0.767, 0.843], folds 0.811 ± 0.044; balanced accuracy 0.428 [0.354, 0.499]; 7 s.
- vs row 5: AUROC gain +0.007 [-0.011, +0.023], wins in 78% of resamples.
- vs best so far (s0-T005): AUROC gain +0.006 [-0.004, +0.016], wins in 87% of resamples.

## s0-T007 Experiment: L3, phase 1

Why: Phase 1 screening, recipe 7 of 11, fixed values, fixed order

Macro AUROC 0.719 [0.672, 0.767], folds 0.721 ± 0.063; balanced accuracy 0.426 [0.355, 0.497]; 26 s.
- vs row 5: AUROC gain -0.080 [-0.118, -0.043], wins in 0% of resamples.
- vs best so far (s0-T006): AUROC gain -0.088 [-0.122, -0.054], wins in 0% of resamples.

## s0-T008 Experiment: L4, phase 1

Why: Phase 1 screening, recipe 8 of 11, fixed values, fixed order

Macro AUROC 0.819 [0.780, 0.854], folds 0.831 ± 0.031; balanced accuracy 0.477 [0.402, 0.555]; 26 s.
- vs row 5: AUROC gain +0.019 [-0.015, +0.053], wins in 85% of resamples.
- vs best so far (s0-T006): AUROC gain +0.012 [-0.020, +0.042], wins in 77% of resamples.

## s0-T009 Experiment: N1, phase 1, size 32

Why: Phase 1 screening, recipe 9 of 11, fixed values, fixed order

Macro AUROC 0.754 [0.711, 0.796], folds 0.740 ± 0.045; balanced accuracy 0.344 [0.278, 0.412]; 7 s.
- vs row 5: AUROC gain -0.045 [-0.077, -0.015], wins in 0% of resamples.
- vs best so far (s0-T008): AUROC gain -0.064 [-0.102, -0.025], wins in 0% of resamples.

## s0-T010 Experiment: N2, phase 1, size 32

Why: Phase 1 screening, recipe 10 of 11, fixed values, fixed order

Macro AUROC 0.757 [0.713, 0.799], folds 0.758 ± 0.058; balanced accuracy 0.388 [0.317, 0.457]; 8 s.
- vs row 5: AUROC gain -0.042 [-0.075, -0.014], wins in 0% of resamples.
- vs best so far (s0-T008): AUROC gain -0.062 [-0.101, -0.025], wins in 0% of resamples.

## s0-T011 Experiment: N3, phase 1, size 32

Why: Phase 1 screening, recipe 11 of 11, fixed values, fixed order

Macro AUROC 0.785 [0.743, 0.823], folds 0.783 ± 0.083; balanced accuracy 0.400 [0.326, 0.474]; 6 s.
- vs row 5: AUROC gain -0.014 [-0.045, +0.013], wins in 16% of resamples.
- vs best so far (s0-T008): AUROC gain -0.033 [-0.071, +0.003], wins in 4% of resamples.

## s0-D001 Decision: shortlist

Best in Phase 1: s0-T008 (L4, AUROC 0.819). A recipe stays in if the best does not win in at least 90% of paired resamples; 2 tied, keeping at most 3: s0-T008 (L4, AUROC 0.819), s0-T006 (L2, AUROC 0.807), s0-T005 (L1, AUROC 0.801).

Based on: s0-T001, s0-T002, s0-T003, s0-T004, s0-T005, s0-T006, s0-T007, s0-T008, s0-T009, s0-T010, s0-T011

- s0-T001: best_wins_share 0.955, gain 0.029, recipe E1
- s0-T002: best_wins_share 0.961, gain 0.030, recipe E2
- s0-T003: best_wins_share 1.000, gain 0.072, recipe E3
- s0-T004: best_wins_share 1.000, gain 0.072, recipe E4
- s0-T005: best_wins_share 0.871, gain 0.018, recipe L1
- s0-T006: best_wins_share 0.768, gain 0.012, recipe L2
- s0-T007: best_wins_share 1.000, gain 0.100, recipe L3
- s0-T009: best_wins_share 0.999, gain 0.064, recipe N1
- s0-T010: best_wins_share 0.999, gain 0.062, recipe N2
- s0-T011: best_wins_share 0.960, gain 0.033, recipe N3

Action: `{"phase1_ids": ["s0-T008", "s0-T006", "s0-T005"], "shortlist": ["L4", "L2", "L1"]}`

## s0-T012 Experiment: L4, phase 2

Why: Phase 2: tune L4 (C by inner CV); it has no size (based on s0-D001)

Macro AUROC 0.797 [0.760, 0.830], folds 0.822 ± 0.025; balanced accuracy 0.451 [0.378, 0.524]; 414 s.
- vs row 5: AUROC gain -0.002 [-0.041, +0.035], wins in 45% of resamples.
- vs best so far (s0-T008): AUROC gain -0.022 [-0.051, +0.008], wins in 8% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'stack': 0.01, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 1.0, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.001, 'stack': 0.01, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 0.001, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'genbio_pathfm': 0.01, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'genbio_pathfm': 0.01, 'stack': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 1.0, 'stack': 0.1, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T013 Experiment: L2, phase 2

Why: Phase 2: tune L2 (C by inner CV); it has no size (based on s0-D001)

Macro AUROC 0.797 [0.755, 0.838], folds 0.801 ± 0.021; balanced accuracy 0.429 [0.357, 0.501]; 327 s.
- vs row 5: AUROC gain -0.002 [-0.028, +0.020], wins in 44% of resamples.
- vs best so far (s0-T008): AUROC gain -0.022 [-0.058, +0.015], wins in 13% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-T014 Experiment: L1, phase 2

Why: Phase 2: tune L1 (C by inner CV); it has no size (based on s0-D001)

Macro AUROC 0.796 [0.756, 0.835], folds 0.802 ± 0.051; balanced accuracy 0.429 [0.357, 0.501]; 331 s.
- vs row 5: AUROC gain -0.003 [-0.020, +0.013], wins in 38% of resamples.
- vs best so far (s0-T008): AUROC gain -0.022 [-0.055, +0.011], wins in 10% of resamples.
- Chosen per fold: [{'M': 0.1, 'genbio_pathfm': 0.1, 'uni2h': 0.001, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.0001, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 0.001}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.001, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 0.001, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.01, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 0.001, 'genbio_pathfm': 0.01, 'uni2h': 0.01, 'virchow2': 0.01}, {'M': 1.0, 'genbio_pathfm': 1.0, 'uni2h': 0.01, 'virchow2': 1.0}]

## s0-D002 Decision: final_selection

s0-T012 (L4, AUROC 0.797) has the highest AUROC but does not clearly beat the simpler s0-T014 (L1, AUROC 0.796): it wins in 50% of resamples (needs ≥ 90%) and its balanced accuracy differs by +0.022 (may be at most 0.02 lower); the simpler recipe is chosen.

Based on: s0-T012, s0-T013, s0-T014, s0-D001

- s0-T014: bacc_difference 0.022, candidate_wins_share 0.502, clearly_beaten False, gain 0.001, recipe L1

Action: `{"candidate": "s0-T012", "winner": "s0-T014", "winner_phase": 2, "winner_recipe": "L1", "winner_size": null}`
