# Decisions: the version log of the fusion-search agent

This folder records every version of the fusion-search agent (TASK.md Step 2): what each
version was, why it was built, what it found, and what it led to. It is the decision log at
the level of designs; each version's experiment-level decision log is in
`artifacts/versions/vNN/seed_0/agent/decision_log.md`.

The loop that produces these versions is specified in [`../docs/loop.md`](../docs/loop.md).

## Files

| File | Content |
| --- | --- |
| `README.md` | This index and the version leaderboard |
| `TEMPLATE.md` | The required structure of every version file |
| `vNN.md` | One file per version (two digits: `v01`, `v02`, …) |
| `SUMMARY.md` | Written once, when the loop ends |

## Rules

1. A version file is written in two stages. **Before any run**: Motivation, Design,
   Pre-registration. **After the run**: Results, Analysis, Step back, Proposal for the next
   version. Text written before the run is never edited afterwards; corrections are added
   as dated amendments.
2. Every number is copied from the version's `trials.jsonl` or `summary.csv`, never retyped
   from memory.
3. Every version is reproducible from the shared code and its config (`conf/versions/vNN.yaml`):
   earlier versions' code paths are never changed, and their tests keep passing.
4. Development data of seed 0 only. The test set is never used.

## Leaderboard (seed 0, development data, macro AUROC out-of-fold over 162 patients)

Row 5 (best single model + age and sex, plain concatenation) is the reference.
"Current best" changes only when a version's winner clearly beats the current best
(wins in ≥ 90% of the paired resamples, balanced accuracy at most 0.02 lower).

| Version | Tag | Models | Key idea | Experiments | Winner | AUROC [95% CI] | Bal. acc. | vs row 5: gain [95% CI], share won | Winner's curse | Current best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| row 5 | — | U, V | reference | — | UNI2-h + age, sex | 0.799 | 0.456 | — | — | — |
| v01 | `v01` | U, V | Fixed sweep → shortlist → tune | 13 / 20 | L2 (tuned) | 0.797 [0.754, 0.836] | 0.447 | −0.002 [−0.025, +0.016], 43% | +0.009 | yes (v01–v03) |
| v02 | `v02` | U, V | Adaptive loop; age-and-sex tokenizers; matched baselines | 15 / 20 | L1:raw (screen) | 0.799 [0.757, 0.838] | 0.414 | −0.000 [−0.023, +0.020], 49% | +0.010 (global, 28) | no (v01 stays) |
| v03 | `v03` | U, V | Age and sex as a Bayesian prior (log-linear pools) | 20 / 20 | L1:raw (screen) | 0.799 [0.757, 0.838] | 0.414 | −0.000 [−0.023, +0.020], 49% | +0.011 (global, 48) | no (v01 stays) |
| v04 | `v04` | U, V | Tile-level MIL fusion (gated attention over cached tiles) | 17 / 20 | M1:ft (screen) | 0.854 [0.818, 0.885] | 0.546 | +0.054 [+0.022, +0.084], 99.9% | +0.003 (global, 65) | **yes** |
| v05 | `v05` | U, V | Separately trained MIL experts × age-and-sex prior (log-linear pools) | 20 / 20 | M1:ft (screen) | 0.854 [0.818, 0.885] | 0.546 | +0.054 [+0.022, +0.084], 99.9% | +0.005 (global, 83) | no (v04 stays) |

U = UNI2-h, V = Virchow2, G = GenBio-PathFM.

**Block 2: U, V, G** (from 2026-11-02, after GenBio-PathFM; docs/loop.md). Row 5 on seed 0 is
unchanged (UNI2-h + age and sex, 0.799). Re-baselines: v01 and the current best (v04) re-run with
their configs on three models (`conf/versions/v01_uvg.yaml`, `v04_uvg.yaml`). Winner's curse is
global over the experiments of this block.

| Version | Tag | Models | Key idea | Experiments | Winner | AUROC [95% CI] | Bal. acc. | vs row 5: gain [95% CI], share won | Winner's curse | Current best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v01_uvg | — | U, V, G | v01 re-baseline | 14 / 20 | L1 (tuned) | 0.796 [0.756, 0.835] | 0.429 | −0.003 [−0.020, +0.013], 38% | +0.008 (14) | yes |
| v04_uvg | — | U, V, G | v04 re-baseline | 17 / 20 | M2:ft (screen) | 0.851 [0.813, 0.886] | 0.502 | +0.052 [+0.014, +0.084], 99.6% | +0.003 (28) | **yes** |
| v06 | `v06` | U, V, G | Composition pooling (tile class distributions averaged) | 20 / 20 | M2:ft (screen) | 0.851 [0.813, 0.886] | 0.502 | +0.052 [+0.014, +0.084], 99.6% | +0.005 (46) | no (v04_uvg stays) |
| v07 | `v07` | U, V, G | CLAM experts (instance clustering loss) in the product with age and sex | 20 / 20 | X1:raw (screen) | 0.858 [0.825, 0.889] | 0.520 | +0.059 [+0.029, +0.085], 99.95% | +0.004 (66) | no (X1 vs v04_uvg M2: 83%) |
| v08 | `v08` | U, V, G | Spatial-context tokens (3×3, 7×7 neighbourhood means) | 20 / 20 | X1:raw (screen) | 0.858 [0.825, 0.889] | 0.520 | +0.059 [+0.029, +0.085], 99.95% | +0.004 (86) | no |
| v09 | `v09` | U, V, G | Ensembles of expert types (product of experts) | 20 / 20 | X1:raw (screen) | 0.858 [0.825, 0.889] | 0.520 | +0.059 [+0.029, +0.085], 99.95% | +0.004 (106) | no |
| v10 | `v10` | U, V, G | Age and sex as attention queries over tiles | 20 / 20 | X1:raw (screen) | 0.858 [0.825, 0.889] | 0.520 | +0.059 [+0.029, +0.085], 99.95% | +0.006 (126) | no |
| v11 | `v11` | U, V, G | Auxiliary grade-group target for the experts | 20 / 20 | **H1:raw (screen)** | **0.863 [0.831, 0.894]** | 0.521 | +0.064 [+0.033, +0.093], 100% | +0.003 (146) | **yes** (vs v04_uvg M2: 97.4%, bal. acc. +0.019) |

The loop ended after v11 (version budget); summary: [`SUMMARY.md`](SUMMARY.md). Test results of the
frozen designs (H1 primary, H2, C1) on seeds 0–4: [`../docs/results.md`](../docs/results.md).

v06 (post-hoc): X1:raw 0.858 is the highest AUROC on U, V, G; vs UNI2-h expert alone 77%, vs U + V +
G without age and sex 92.5%, vs UNI2-h + age and sex 59%. v07 (post-hoc): C1:raw 0.861 (highest so far); vs UNI2-h
CLAM expert alone 80%, vs no age and sex 89%, vs UNI2-h + age and sex 60%.

**Matched rows (fair comparison, same recipe and tuning as the winner).** v04: UNI2-h alone 0.853,
Virchow2 alone 0.841, UNI2-h + age and sex 0.851, U + V without age and sex 0.857, winner 0.854:
the v04 gain over row 5 comes from the tile-level representation; fusion and age and sex add ≈ 0.
v05 (post-hoc, best AUROC candidate X1:raw 0.858): UNI2-h expert alone 0.853 (X1 wins 79%), U + V without age and sex
0.855 (83%), UNI2-h + age and sex 0.856 (60%).
