# vNN: <one-line name of the version>

Status: designed | implemented | run | analysed. Date(s). Git tag `vNN` (commit `<hash>`).
Config: `conf/versions/vNN.yaml`. Outputs: `artifacts/versions/vNN/seed_0/agent/`.
Models: <U, V | U, V, G>. Previous version: `vNN-1`.

<!-- Sections 1–4 are written BEFORE the run and never edited afterwards (add dated
amendments at the end instead). Sections 5–9 are written AFTER the run. -->

## 1. Motivation

What the previous version showed (cite trial IDs and numbers) and which bottleneck this
version attacks. One paragraph.

## 2. Design

The one structural change, stated exactly: what is added, removed or replaced (search
space, representation, age-and-sex encoding, objective, training data, search policy or
selection rule). What is held fixed. Equations or shapes where they help.

## 3. Pre-registration

- **Hypothesis:** what should happen and why (mechanism).
- **Prediction:** the expected direction and rough size of the effect.
- **Success criterion:** e.g. "the winner clearly beats the current best version" or
  "the metadata diagnostic shows a gain whose CI excludes 0".
- **Budget:** experiments (≤ 20), cap per experiment.
- **What would falsify it.**

## 4. Implementation

Files added or changed, tests added, the exact run commands.

## 5. Results

Copied from `summary.csv` / `trials.jsonl`:
- table of every experiment (ID, candidate, phase or stage, AUROC [CI], balanced accuracy,
  gain vs row 5 [CI], share won, seconds);
- every decision (ID, rule, statistics, action);
- winner, matched baselines (rows 1–5 computed with the winner's recipe and tuning),
  metadata diagnostic, winner's-curse estimate, run time;
- leaderboard row (also added to `README.md`).

## 6. Analysis

What worked and what did not, with evidence (trial IDs, numbers, CIs). Per-class AUROC and
confusion of the winner versus row 5. Was the hypothesis supported? Did the feedback loop
react to evidence (count the decisions that changed the plan)?

## 7. Step back (first principles)

1. **Bottleneck now:** signal (information not in the inputs), bias (model too simple),
   variance (too little data for the model), selection noise (differences below what 162
   patients can resolve), or search (good candidates never tried)? Give the evidence.
2. **Assumptions that may be wrong** in the current design.
3. **Directions** (at least three), each with: mechanism, why it attacks the bottleneck,
   reference(s), cost, risk of overfitting the validation data.

## 8. Proposal for vNN+1

The chosen structural change and why it beats the alternatives above. What stays fixed.

## 9. Amendments

Dated corrections, if any.
