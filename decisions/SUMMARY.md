# Summary of the version loop (v01–v11)

Written 2026-11-02, when the loop ended after v11 (the version budget set by the project lead). All numbers are seed 0, development data (162 patients, 8-fold out-of-fold), macro
AUROC; the test set was not read during the loop. Per-version detail: `vNN.md`; leaderboard:
`README.md`.

## 1. What was searched

Each version is one structural change, pre-registered before its run, searched by the
deterministic inner agent (≤ 20 experiments, 30-minute cap each, paired-bootstrap decisions),
and analysed by the outer language-model layer (docs/loop.md). 11 versions + 2 re-baselines,
236 experiments. v01–v05: UNI2-h + Virchow2; v06–v11: UNI2-h, Virchow2, GenBio-PathFM.

| Version | Structural change | Best candidate (AUROC) | Agent's winner | Outcome |
| --- | --- | --- | --- | --- |
| v01 | Fixed sweep of 11 recipes (early, late, learned), shortlist, tune | L4 tuned 0.816 (bal. acc. low) | L2 tuned 0.797 | Below row 5 |
| v02 | Adaptive policy; learned age-and-sex tokens | L1:raw 0.799 | L1:raw 0.799 | Encodings add nothing |
| v03 | Age and sex as a Bayesian prior (log-linear pools) | P1:raw 0.807 | L1:raw 0.799 | Ceiling ~0.81 on mean-pooled vectors |
| v04 | Tile-level attention MIL (1,024 tiles per slide) | M1:ft 0.854 | M1:ft 0.854 | **+0.055**: the main gain |
| v05 | One MIL expert per model × age-and-sex expert (product of experts) | X1:raw 0.858 | M1:ft 0.854 | Product best fusion form; fitted weights lose |
| v06 | GenBio-PathFM added; composition pooling | X1:raw 0.858 | M2:ft 0.851 | Falsified: relevance weighting is the gain |
| v07 | CLAM experts (pseudo-labelled tiles) | C1:raw 0.861 | X1:raw 0.858 | Small, unresolved gain |
| v08 | Spatial context tokens (neighbourhood means) | C1:raw 0.861 | X1:raw 0.858 | Falsified: capacity hurts |
| v09 | Ensembles of expert types | C1:raw 0.861 | X1:raw 0.858 | Falsified: experts share errors |
| v10 | Age and sex as attention queries | C1:raw 0.861 | X1:raw 0.858 | Falsified: no better than a prior |
| v11 | Grade-group auxiliary target | H2:raw 0.866 | **H1:raw 0.863** | Supported (vs X1 96%) |

## 2. Current best and the final design

**Current best: v11, winner H1:raw** (AUROC 0.863 [0.831, 0.894], balanced accuracy 0.521). It
clearly beats the previous current best on U, V, G (`v04_uvg`, M2:ft 0.851): gain +0.013
[−0.000, +0.028], won in 97.4% of resamples, balanced accuracy +0.019.

**H1:raw, the agent-selected design:** per foundation model (UNI2-h, Virchow2, GenBio-PathFM),
a gated-attention MIL expert over 1,024 cached tiles (projection to d = 32, 100 epochs, AdamW,
weight decay 0.01, dropout 0.25, 256 tiles per step, 3 seeds averaged), trained with a
seven-class head and an auxiliary grade-group head (In situ; Lepidic; Acinar + Papillary;
Solid + Micropapillary + Cribriform; weight 0.5). No age and sex inside the experts. Fusion:
`z = mean over models of log p_model + log p(class | age, sex)`, the latter from a
class-weighted logistic regression on raw age and sex (C = 1). Zero fitted fusion weights.

**Matched rows of H1:raw (same recipe, seed 0):** age and sex alone 0.611; UNI2-h 0.856;
Virchow2 0.854; GenBio-PathFM 0.841; UNI2-h + age and sex 0.859; U + V + G without age and sex
0.860. Gain over row 5 (0.799): +0.064 [+0.033, +0.093] (from the record of T016).

**Global winner's curse:** +0.003 AUROC over the 146 experiments of the U, V, G block (README.md),
smaller than the gain over row 5 but as large as the gains of v05–v11 over one another.

## 3. What was learned

1. **The gain over row 5 comes from the slide representation.** Weighting tiles by a learned
   relevance (attention MIL) instead of averaging them: 0.80 → 0.85 (v04). Uniform averages,
   in embedding or probability space, stay at ~0.80 (v06).
2. **Age and sex add 0 to +0.005 in every form** (columns, piecewise-linear and learned
   tokens, prior, FiLM, attention queries). Only the product of experts makes the gain
   consistent (89–94% of resamples). They carry little beyond the image (alone 0.61–0.63).
3. **The three foundation models are largely redundant.** Fitted weights ≤ 0.1 for Virchow2
   and GenBio-PathFM next to UNI2-h; joint networks get worse when a model is added.
4. **With ~142 training slides, added capacity loses** (v06, v07 class-specific attention,
   v08); zero-parameter combinations win; tuning never helped the tile-level models.
5. **All experts share their errors** (v09), concentrated on Lepidic, Acinar and Papillary;
   the auxiliary grade-group target (v11) is the only later change that helped beyond its
   control.

## 4. Recommended final design and secondary designs

Frozen in `conf/final.yaml` before the test set is read:

| Role | Design | Seed 0 dev AUROC | Why |
| --- | --- | --- | --- |
| **Primary** | **H1:raw** (v11) | 0.863 | The agent's own selection; current best |
| Secondary | H2:raw (v11) | 0.866 | Highest development AUROC |
| Secondary | C1:raw (v07) | 0.861 | Best design without the grade-group head |

All three are reported on every seed whatever they score; the primary is never changed after
the test numbers are seen.

## 5. Limits

- Every number above is from seed 0 development data, on which ~236 experiments and 11 design
  decisions were made; the differences between v05–v11 (≤ 0.008) are of the size of the
  selection bias.
- Seeds 1–4 share development patients with seed 0's test set and vice versa; their test results
  are secondary (docs/methods.md D22).

## 6. Final evaluation (stage 12)

Per seed s ∈ {0, …, 4}, for each design: 8-fold development CV; one fit on all 162 development
patients; one prediction of the 41 test patients. Rows under each design's own recipe: fused +
age and sex, each model alone, best single model (chosen on development) + age and sex, age and
sex alone, fused without age and sex; plus the original rows 1–5 (stage 09). Claim rule (fixed in
advance): fused beats a row on a seed if the paired 95% CI of the test AUROC gain excludes 0.

```
uv run --no-sync python -m scripts.12_final_eval --seed <s> --device cuda:0|cuda:1
uv run --no-sync python -m scripts.13_final_report
```

Results: `docs/results.md` and `artifacts/final/`.
