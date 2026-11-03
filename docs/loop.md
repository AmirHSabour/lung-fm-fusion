# The version loop (outer layer)

The fusion-search agent ([`agent.md`](agent.md)) is improved **version by version** by a coding
agent with a language model (e.g. Claude Code or Codex). This file is that agent's complete
instruction: given to a fresh agent, it reproduces the process. Versions v01–v11 were produced
this way; their records are in [`../decisions/`](../decisions/README.md).

Together the two layers form a **hybrid agent** (TASK.md line 113):
- **Inner layer (deterministic):** within one version, the agent of `src/agent/` proposes,
  runs and selects experiments, ≤ 20 per version. No language model, no key; every version
  re-runs from its config without the outer layer.
- **Outer layer (language model):** between versions, it reads the inner agent's records,
  reasons from first principles and the literature, and designs the next version. Its reasoning
  is logged, pre-registered, in `decisions/vNN.md`. Without an API key or a coding agent, the
  system degrades to the inner layer: any version's search runs unchanged.

**Invocation.** Give a coding agent this file and a target, e.g. "follow docs/loop.md up to v15".
It resumes from the files (§3) and stops after the target version is closed (§10).

## 1. The loop

```
for N = next version … target:
    A. Design      read vN-1's analysis → write decisions/vNN.md §1–4 (before any run) → commit
    B. Implement   code + conf/versions/vNN.yaml + tests; all tests pass → commit
    C. Run         inner agent on seed 0 development data (≤ 20 experiments), matched rows,
                   leaderboard
    D. Record      decisions/vNN.md §5, decisions/README.md leaderboard → commit
    E. Analyse     what worked and why, with evidence → §6
    F. Step back   first principles + state of the art → §7; choose vNN+1 → §8
    G. Close       commit, tag vNN
end: decisions/SUMMARY.md (§10), then stop and ask before any test-set use
```

## 2. Rules

1. **Never use the test set.** During the loop only `src/eval/dataset.load_dev` reads data.
2. **Seed 0 development data only** (162 patients, 8 fixed folds). Seeds 1–4 contain seed 0's test
   patients in their development sets and are reserved for the final evaluation.
3. **Only age and sex** as metadata inputs; never `Pathological_Diagnosis` or other columns.
4. **Never re-run a foundation model**; use the cached embeddings and features.
5. **Never change** the splits, cohort, tiles, embeddings, `conf/data.yaml`, or rows 1–5.
6. **Compute:** fp32, TF32 off; networks on the training GPU (`conf/train.yaml`), scikit-learn on
   CPU with one BLAS thread per fit (methods D18).
7. **Budget:** ≤ 20 experiments per version, 30-minute cap each.
8. **Hyperparameters** by inner cross-validation or fixed in the config before the run.
9. **Pre-register:** `decisions/vNN.md` §1–4 are committed before the run and never edited;
   corrections go to §9 Amendments, dated.
10. **Earlier versions stay reproducible:** add new modules and registry entries; never change an
    earlier version's code path; every earlier version's tests keep passing.
11. **No language model inside the code, no keys or secrets in the repository.**
12. **Cite only references you can verify**; mark the others "unverified".
13. **Report every version**, including failures; never re-run an experiment to improve a number.
14. **Ask the human before** any test-set use, deleting files, adding an external model or remote
    code, or a dependency with a non-standard licence.
15. Run commands with `uv run --no-sync` (plain `uv run` reinstalls `flash-attn`).

## 3. Where you are

The state is defined by files, never by memory. The last row of the `decisions/README.md`
leaderboard is the last finished version; `git tag --list 'v*'` lists closed versions; the
sections filled in the highest `decisions/vNN.md` give the step (§1–4 only: B or C; §5: E; §8
without a tag: G). An interrupted run resumes by re-issuing its command; if the code changed
since it started, it stops: commit, delete that version's output directory, re-run, and record
this in §9 Amendments.

## 4. Shared protocol (identical for every version)

Versions may change the search space, the representation, the age-and-sex encoding, the
objective, the training-data construction (from development patients only), the policy or the
selection rule. They may not change:

- **Data:** seed 0 development set; inputs are the cached representations and age and sex.
- **One experiment:** 8-fold out-of-fold prediction; macro AUROC (primary), balanced accuracy,
  per-class AUROC, confusion matrix, CIs; paired comparisons on the stored 2,000 resamples.
- **"Clearly beats":** wins in ≥ 90% of the paired resamples and balanced accuracy at most 0.02
  lower.
- **Records:** append-only `trials.jsonl` and the reports generated from it ([`agent.md`](agent.md) §5).
- **After every run:** matched baselines and the metadata diagnostic (`--matched`), and the
  leaderboard with the global winner's curse (`scripts/version_leaderboard.py`).
- **Outputs:** `conf/versions/vNN.yaml` (with `out_dir: artifacts/versions/vNN`) and
  `artifacts/versions/vNN/seed_0/agent/`.

## 5. One iteration (`decisions/vNN.md`, from [`TEMPLATE.md`](../decisions/TEMPLATE.md))

**A. Design.** §1 Motivation from vN-1 §6–8, with trial IDs and numbers. §2 Design: exactly
**one structural change**, stated precisely, with what is held fixed. §3 Pre-registration:
hypothesis, mechanism, predictions, success criterion, falsifier, budget. §4 Implementation
plan. Commit `vNN: design`.

**B. Implement.** New modules and registry entries by name; a new policy module only if the
policy changes; `conf/versions/vNN.yaml` from the previous version's; tests for every new
component (shapes, training-rows-only fitting, determinism, caching, config). All tests pass.
Commit `vNN: implementation`.

**C. Run.**
```bash
uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/vNN.yaml
uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/vNN.yaml --matched
uv run --no-sync python -m scripts.version_leaderboard
```
Post-hoc rows for any experiment: `scripts/11_post_hoc.py --of <trial id>` (analysis only).

**D. Record.** §5 Results from `summary.csv` and `trials.jsonl`: every experiment and decision,
the winner, matched rows, metadata diagnostic, comparison with the current best and row 5,
winner's curse. Add the leaderboard row; the current best changes only if the new winner clearly
beats it. Commit `vNN: results`.

**E. Analyse (§6).** Was the hypothesis supported (numbers and CIs)? Which mechanism explains
the result? Failure analysis of the winner (per-class AUROC, confusions). Did the inner loop react
to evidence?

**F. Step back (§7–8).** See §6 below. **G. Close.** Commit, `git tag -a vNN`.

## 6. Stepping back between versions

Answer in writing, each with evidence:
1. **What limits performance now?** Signal (the information is not in the inputs), bias (the
   model cannot express the function), variance (too many parameters for ~142 training
   patients), selection noise (differences below what 162 patients resolve), search (promising
   candidates untried), or objective mismatch.
2. **Which assumptions of the current design could be wrong,** and how would each be tested?
3. **What does the literature do about this bottleneck?** At least three directions, each with
   mechanism, the bottleneck it attacks, a reference, cost, and risk of overfitting.
4. **Choose one direction** and argue why it dominates now.

A **structural change** alters what is searched or how: the fusion family, the slide
representation (pooling, tile-level models), the age-and-sex encoding, the objective, the data
construction, the policy or the selection rule. A new grid value, seed or single hyperparameter
is not a version. Prefer changes whose expected benefit exceeds the resolution limit (≈ 0.01–0.03
AUROC at n = 162).

**Directions catalogue** (use only when the analysis points to them): tile-to-slide aggregators
(ABMIL, CLAM, TransMIL, DSMIL, Perceiver-style cross-attention, top-k, prototype pooling);
age-and-sex tokens interacting with tiles; FiLM conditioning; feature tokenizers and numeric
embeddings; tensor or low-rank fusion; mixtures of experts; ensemble selection; temperature
scaling; logit adjustment; hierarchical targets built from the label; successive halving or
Bayesian optimisation of the search; shrinkage or prototype classifiers for small n.

## 7. Guarding against overfitting across versions

Many versions on the same 162 patients overfit unless: every version is pre-registered;
hyperparameters come from inner CV; a version becomes current best only if it clearly beats the
current best; the global winner's curse is reported after every version (say plainly when it is
comparable to the gain); seeds 1–4 stay unused; and only the final test evaluation supports a claim.

## 8. Adding a foundation model

When a new model's embeddings become available: summarise and check the manifest
(`07_extract_embeddings --summarize`), build features and tile caches (stages 08, 08b), recompute
rows 1–5 (stage 09), then re-run v01 and the current best version on the new model set and start
a new block of the leaderboard (as `v01_uvg`, `v04_uvg` did when GenBio-PathFM was added).

## 9. When something goes wrong

Failing tests are fixed before any run, never skipped. A failed or timed-out experiment counts
against the budget and is analysed. A bug found after a run is fixed, the version re-run in a
fresh directory, and both recorded in §9 Amendments. A version that cannot be implemented is
recorded as "not run" with the reason.

## 10. Ending the loop

When the target version is closed, write `decisions/SUMMARY.md`: the leaderboard, the current
best version and its winner, its matched rows, its gain over row 5, the global winner's curse,
what was learned, and the recommended final designs. Freeze the final designs in
`conf/final.yaml` (primary = the current best's winner). Then **stop and ask** before the final
evaluation, which reads the test sets once (`scripts/12_final_eval.py`).
