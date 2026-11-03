# Fusing pathology foundation models with age and sex for lung adenocarcinoma subtyping

We predict the histologic growth pattern (7 classes) of lung adenocarcinoma from whole-slide images
of the CLWD dataset ([task](docs/TASK.md)) by fusing three pathology foundation models — UNI2-h,
Virchow2 and GenBio-PathFM — with patient age and sex. A **hybrid agent** searches how to fuse them:
a deterministic inner agent runs a budgeted, auditable search within each version, and a
language-model outer layer designs the next version from the inner agent's records. Over 11
pre-registered versions (236 experiments, development data only), the agent moved from mean-pooled
slide vectors to tile-level attention experts, multiplied with an age-and-sex prior. On five
held-out test sets the selected design reaches a macro AUROC of **0.894 ± 0.012**, against 0.835 ±
0.035 for the best single model plus age and sex as originally specified.

## Results

Test set, primary design (H1, agent-selected), mean ± sd over 5 splits (seeds 0–4, 41 test patients
each). *Same recipe*: every row uses the fused design's tile-level classifier (TASK.md line 162).
*Original*: mean-pooled slide vectors and logistic regression.

| Configuration | Macro AUROC (same recipe) | Balanced acc. (same recipe) | Macro AUROC (original) | Balanced acc. (original) |
| --- | --- | --- | --- | --- |
| Metadata only (age + sex) | 0.594 ± 0.047 | 0.275 ± 0.031 | 0.594 ± 0.047 | 0.275 ± 0.031 |
| UNI2-h alone | 0.893 ± 0.013 | 0.549 ± 0.123 | 0.844 ± 0.032 | 0.487 ± 0.081 |
| Virchow2 alone | 0.881 ± 0.011 | 0.544 ± 0.051 | 0.831 ± 0.033 | 0.475 ± 0.109 |
| GenBio-PathFM alone | 0.884 ± 0.015 | 0.538 ± 0.062 | 0.844 ± 0.030 | 0.479 ± 0.060 |
| Best single model + metadata | 0.895 ± 0.011 | 0.561 ± 0.099 | 0.835 ± 0.035 | 0.486 ± 0.062 |
| **Fused (agent-selected) + metadata** | **0.894 ± 0.012** | **0.560 ± 0.045** | | |

The fused model exceeds every original single-model row on every split (paired 95% CI above zero
on 4–5 of 5 splits). Under the identical tile-level recipe it exceeds Virchow2 and GenBio-PathFM on
all splits but only matches UNI2-h (+0.002); age and sex add +0.002. The gain over the original rows
therefore comes from tile-level attention, not from fusion. The weakest classes are Lepidic and
Papillary, mostly predicted as Acinar. Full tables with per-split bootstrap CIs, paired comparisons,
per-class AUROC, confusion matrices and a sensitivity analysis: [`docs/results.md`](docs/results.md).

## Method

| Stage | Summary | Details |
| --- | --- | --- |
| Data | 203 patients with images (6 without a JPG excluded); one slide per patient (lowest WSI number); duplicate ID merged; conflicting labels resolved per selected slide | [methods §1–2](docs/methods.md) |
| Splits | Per seed 0–4: stratified 41-patient test set; 8-fold CV on the 162 development patients; committed | [methods §3](docs/methods.md#3-splits) |
| Tiles | True pixel size measured per slide, resampled to 0.5 µm/px (20×); tissue and pen-ink masks; 224-px tiles, stride 224 | [methods §4](docs/methods.md#4-slides-and-tiles) |
| Embeddings | Each model's own preprocessing; every tile embedded once (fp32) and cached | [methods §5](docs/methods.md#5-foundation-models-and-embeddings) |
| Baselines | Rows 1–5: logistic regression with C by inner CV, identical for every row | [methods §7](docs/methods.md#7-baselines-rows-15-of-taskmd-53) |
| Agent | Hybrid: deterministic inner search (≤ 20 experiments per version) + language-model outer loop (11 versions) | [agent](docs/agent.md), [loop](docs/loop.md), [decisions](decisions/README.md) |
| Final evaluation | Three designs frozen before any test set was read; each trained once per split and tested once | [results](docs/results.md) |

**The hybrid agent.** The *inner layer* (`src/agent/`) proposes an experiment, scores it by 8-fold
cross-validation, compares it with the best so far by a paired bootstrap, and decides the next move
— opening each fusion family, following the leading one, dropping families that are clearly behind,
tuning the best by inner CV, and selecting the simplest candidate the best does not clearly beat.
Every step is logged with its rule and reason. The *outer layer* follows [`docs/loop.md`](docs/loop.md):
between versions it analyses the records, reasons from first principles and the literature, and
pre-registers one structural change per version in [`decisions/`](decisions/README.md). Overfitting
is controlled by the budget, inner-CV tuning, the 90% paired-bootstrap rule, a winner's-curse
estimate (+0.003 AUROC over 146 experiments), and seed 0 development data only.

**Final design (H1).** Per foundation model, a gated-attention multiple-instance expert over 1,024
tiles per slide, trained with an auxiliary grade-group target; the three experts' log-probabilities
are averaged and added to the log-probabilities of a logistic age-and-sex model. No fitted fusion
weights.

## Reproducing

**Setup.** Python 3.12, a CUDA GPU, [`uv`](https://docs.astral.sh/uv/).
```bash
uv sync                                   # exact environment from uv.lock
uv run --no-sync hf auth login            # UNI2-h and Virchow2 are gated: request access on Hugging Face
```
Use `uv run --no-sync` for every command (plain `uv run` reinstalls `flash-attn`). Our results were
produced on NVIDIA RTX 4090 GPUs (training) and RTX PRO 6000 GPUs (embeddings) in fp32; other GPUs run
the same code, but numbers may differ in the last digits.

**Pipeline.** Each stage is resumable and writes the input of the next.

| Stage | Command (`uv run --no-sync python -m …`) | Output |
| --- | --- | --- |
| 00–01 Models, slides | `scripts.00_download_models`, `scripts.01_download_dataset` | `models/`, `data/raw/` (386 JPGs, 414 GB) |
| 02–04 Scale, cohort, splits | `scripts.02_measure_slide_scale`, `scripts.03_make_cohort`, `scripts.04_make_splits` | `artifacts/` (committed) |
| 05–06 Resample, tiles | `scripts.05_resample_slides`, `scripts.06_tile_slides` | `data/processed/`, `artifacts/tiles/` |
| 07 Embeddings | `scripts.07_extract_embeddings --model <m> --device cuda:0`, then `--summarize` | `artifacts/embeddings/` (≈ 31 GPU-h) |
| 08 Slide features, tile caches | `scripts.08_slide_features`, `scripts.08b_tile_cache --models uni2h virchow2 genbio_pathfm`, `scripts.08c_tile_context` | `artifacts/features/` |
| 09 Baselines | `scripts.09_baselines` | `artifacts/runs/` |
| 10 Agent, one version | `scripts.10_fusion_search --seed 0 --dev --config conf/versions/vNN.yaml [--matched]` | `artifacts/versions/vNN/` |
| Version leaderboard | `scripts.version_leaderboard` | `artifacts/versions/leaderboard_seed0.csv` |
| 11 Post-hoc rows | `scripts.11_post_hoc --seed 0 --config conf/versions/vNN.yaml --of <trial id>` | `posthoc.jsonl` |
| 12 Final evaluation (reads test) | `scripts.12_final_eval --seed <s> --device cuda:0` (one seed per 24-GB GPU) | `artifacts/final/` |
| 13 Results tables | `scripts.13_final_report` | `docs/results.md` |

**Re-running a recorded version** writes to a fresh directory, leaving the committed record intact:
`scripts.10_fusion_search --seed 0 --dev --config conf/versions/v11.yaml --out-dir artifacts/rerun/v11`.

**Continuing the loop.** Give a coding agent [`docs/loop.md`](docs/loop.md) and a target version
(e.g. "follow docs/loop.md up to v15"); it designs, implements, runs and records each version in
`decisions/`.

**Notebooks.** [`01_results.ipynb`](notebooks/01_results.ipynb): the results with figures, and a
re-run of the final evaluation (one split by default; all five by uncommenting one line), checked
against the committed numbers. [`02_agent_loop.ipynb`](notebooks/02_agent_loop.ipynb): the versions,
their decisions and experiments. Both need the cached features (`artifacts/features/`, not
redistributed); on Colab or Kaggle, copy them in first.

**Tests.** `uv run --no-sync pytest` (157 tests: cohort and splits rebuilt byte for byte, split
invariants, preprocessing, every fusion recipe, agent rules and resumption, no test-set leakage).

## Repository

```
conf/          data, models, preprocessing, training; versions/vNN.yaml (one per version); final.yaml
src/           data (cohort, splits, tiling) · embeddings · fusion (recipes by name) · agent · eval
scripts/       one entry point per stage (00–13); experiments/ (side checks E1, E2)
decisions/     the outer layer's record: one pre-registered file per version, leaderboard, summary
docs/          TASK.md · methods.md · agent.md · loop.md · results.md
notebooks/     results and agent walkthroughs
tests/         correctness, reproducibility and leakage tests
artifacts/     committed records: cohort, splits, manifests, baselines, every version's agent log,
               final results (embeddings, features and predictions are git-ignored)
```

Model weights (UNI2-h, Virchow2: CC-BY-NC-ND-4.0; GenBio-PathFM: GenBio AI Community License) and
slides are not redistributed; stages 00–01 download them at pinned revisions. No keys or tokens are
stored in this repository.
