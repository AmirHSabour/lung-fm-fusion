# The fusion-search agent (inner layer)

A structured, deterministic search agent. It proposes a fusion configuration, evaluates it,
reads the result, and conditions its next proposal on everything recorded so far. It uses no
language model and no API key, so every run is reproducible from its configuration. One run of
the agent is one *version* of the search (`conf/versions/vNN.yaml`); the outer layer that designed
versions v01–v11 is described in [`loop.md`](loop.md).

## 1. Task

**Inputs per patient:** the UNI2-h, Virchow2 and GenBio-PathFM representations (U, V, G) and age
and sex (M), on the 162 development patients of seed 0 with their 8 fixed folds. Test patients
are never loaded (`src/eval/dataset.load_dev`).

**Output:** one fused model, row 6 of the TASK.md §5.3 table ("Fused (agent-selected) +
metadata"). Every candidate uses all four sources; candidates differ only in *how* they are
combined, and how age and sex enter is part of each candidate. Rows 1–5 are fixed baselines
computed without the agent (methods D21).

## 2. Search space

A candidate is a **recipe** combined with an **age-and-sex encoding**, written `RECIPE:encoding`
(e.g. `L2:raw`, `N2:ft`). Recipes are registered by name (`src/fusion/registry.py`).

**The TASK.md search space (v01).**

| Family | Recipes | Mechanism |
| --- | --- | --- |
| Early | E1–E4 | Concatenate the blocks, one classifier; ± per-model L2, ± per-model PCA; each block scaled to equal total variance so 2 age-and-sex columns are not drowned by 8,704 image columns (Westerhuis et al., 1998) |
| Late | L1–L4 | One classifier per source, then probability averaging, log-probability averaging, weighted vote (weight = inner-CV AUROC − 0.5), or stacking on inner out-of-fold log-probabilities |
| Learned | N1–N3 | Each block projected to *d* numbers, then gated fusion, self-attention over the streams, or a low-rank bilinear interaction (images × age and sex) |

**Age-and-sex encodings (v02).** Linear families: `raw` (standardised age, sex) and `ple`
(piecewise-linear age, 8 quantile bins; Gorishniy et al., 2022). Learned families: feature
tokenizers `joint`, `ft` (FT-Transformer; Gorishniy et al., 2021), `plr` (periodic) and `ple`.

**Families added by the version loop.** Each version added one family; all stay in the search
space of later versions.

| Version | Family | Recipes | Mechanism |
| --- | --- | --- | --- |
| v03 | pool | P1–P4 | Age and sex as a Bayesian prior: log-linear pool of per-source log-probabilities, fixed or fitted weights |
| v04 | mil | M1–M3 | Gated-attention MIL over 1,024 tiles per slide (Ilse et al., 2018): one network on all models' tiles; age and sex as a learned bias or FiLM |
| v05 | mexp | X1–X4 | One MIL *expert* per model (no age and sex), combined with a logistic age-and-sex expert by a product of experts (X1: zero fitted weights) or fitted pools |
| v06 | agg | A1–A2 | Experts that average tile class distributions (uniform or attention-weighted) |
| v07 | clam | C1–C2 | CLAM experts: attention plus an instance loss on the most and least attended tiles (Lu et al., 2021) |
| v08 | ctx | K1–K2 | Experts on tile tokens extended with 3×3 and 7×7 neighbourhood means |
| v09 | ens | Z1–Z2 | Ensembles of expert types per model |
| v10 | xq, xqm | Q1–Q2 | Cross-attention pooling; Q2 uses age and sex as attention queries |
| v11 | hier | H1–H2 | Experts trained with an auxiliary grade-group target (In situ; Lepidic; Acinar + Papillary; Solid + Micropapillary + Cribriform) |

## 3. One experiment

A candidate is fitted on 7 folds and predicts the 8th, 8 times: one out-of-fold prediction per
development patient. Recorded: macro one-vs-rest AUROC (primary), balanced accuracy, per-class
AUROC, confusion matrix, fold mean ± sd and a 95% bootstrap CI. Every experiment is compared with
the best so far and with row 5 by a **paired, class-stratified bootstrap** over the 162 patients
(2,000 resamples, drawn once per seed and stored, so every comparison uses the same resamples):
gain, its 95% CI, and the share of resamples won (`src/agent/evaluate.py`).

Experiments come in two modes. **Screen:** fixed values set before any result (C = 0.01 for
image classifiers, C = 1 for age-and-sex models and the stacker, weight decay 0.01, size 32).
**Tuned:** every hyperparameter chosen by inner 5-fold CV on the training folds (C or weight decay
from {10⁻⁴, …, 1}; size from {16, 32, 64}), never by the validation score. Tile-level families are
screen-only from v05 (from v07 for `mil`): tuning their networks inside inner folds exceeds the
time cap, and in v04 it did not change their results.

## 4. Policies

The agent is a pure function from the record to the next action (an experiment, a decision, or
stop; `src/agent/policies/`).

### 4.1 v01: two phases (`policy: v01`)

1. **Phase 1, breadth:** E1–E4, L1–L4, N1–N3 once each at the screen values (11 experiments).
2. **Shortlist:** the best Phase 1 recipe *t* plus every recipe that *t* does not beat in ≥ 90% of
   the paired resamples; at most 3.
3. **Phase 2, depth:** each shortlisted recipe tuned by inner CV; recipes with a size try 32, then
   64, then 16 only if 64 was not better (1–3 experiments each).
4. **Final selection** (§4.3).

### 4.2 v02 onwards: adaptive (`policy: v02`)

Phase 1 is replaced by a feedback loop over families; the same policy runs v02–v11 with each
version's search space.
1. **Open:** the simplest candidate of each family (from v07, every recipe of the newest family).
2. **Explore:** after every experiment, *drop* any family (with ≥ 2 experiments) whose best is
   clearly beaten by the best overall; the *leader* is the family in play with the highest best
   AUROC; screen its next untried recipe, then its untried encodings.
3. **Tune:** the three best screened candidates of tunable families, by inner CV.
4. **Final selection** (§4.3).

### 4.3 Final selection (all versions)

Let *c* be the finished experiment with the highest AUROC. *c* **clearly beats** another
experiment if it wins in ≥ 90% of the paired resamples **and** its balanced accuracy is at most
0.02 lower. The winner is the **simplest** experiment that *c* does not clearly beat (simplicity:
family order, then recipe number, then encoding, screen before tuned). Small, noisy gains
therefore never displace a simpler candidate. The winner is frozen in `selected.yaml`.

### 4.4 Budget and stopping

At most **20 experiments** per version, enforced in code; a **30-minute** cap per experiment
(each runs in its own process; a time-out counts against the budget). The search stops at the
final selection, or when the budget is spent (selection is then made on what exists).

## 5. Records

Each run writes `artifacts/versions/vNN/seed_0/agent/`:

| File | Content |
| --- | --- |
| `trials.jsonl` | Append-only record: a header (seed, models, code version, feature checksums, config checksum), then every experiment and decision in order |
| `decision_log.md` | Human-readable log generated from the record: each decision with its rule, statistics and reason; each experiment with why it was chosen and its scores |
| `summary.csv`, `progress.png` | One row per experiment; AUROC ± CI per experiment with row 5 as reference |
| `selected.yaml` | The winner and the winner's-curse estimate |
| `trials/<id>.npz` | Out-of-fold probabilities of each experiment (git-ignored) |

Experiments (`s0-T007`) and decisions (`s0-D002`) cite the IDs they are based on. A
**fingerprint** (hash of recipe, settings and feature checksums) prevents running the same
experiment twice. A run **resumes** from its record after an interruption and refuses to mix code,
data or configuration versions.

## 6. After a run (outside the budget)

- **Matched baselines:** the winner's recipe applied to age and sex alone, to each model alone,
  and to the best single model plus age and sex, so every row shares the winner's classifier
  (TASK.md line 162); plus the **metadata diagnostic** (the winner without age and sex).
- **Leaderboard** (`scripts/version_leaderboard.py`): the winner against the current best version
  (it replaces it only if it clearly beats it) and against row 5, and the **global winner's curse**
  over all experiments of all versions (Tibshirani & Tibshirani, 2009).
- **Post-hoc rows** (`scripts/11_post_hoc.py`): the same comparisons for any recorded experiment,
  written to a separate file; never used for selection.

## 7. Defences against overfitting the validation data

1. Every score uses all 8 folds (162 patients), not one 21-patient validation set.
2. A hard budget of 20 experiments per version.
3. Hyperparameters by inner CV, or fixed before the run.
4. Paired-bootstrap decisions with a 90% rule and a simplicity preference.
5. A winner's-curse estimate per version and across versions (+0.003 AUROC over the 146
   three-model experiments).
6. Seed 0 development data only during the search; designs frozen before any test set is read.
7. Each version pre-registered before its run ([`loop.md`](loop.md)).

## 8. Running it

```bash
uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/v11.yaml            # search
uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/v11.yaml --matched  # matched rows
uv run --no-sync python -m scripts.version_leaderboard                                                        # compare versions
```

v01 can also be run as specified, with the official all-seeds mode and row 5b, through
`conf/agent.yaml` (`--seed <s>`, `--ablation`).
