# Methods

The pipeline and every design decision, numbered D1–D22 (the code cites them by number).
The task is [`TASK.md`](TASK.md); the fusion-search agent is described in [`agent.md`](agent.md)
and the version loop in [`loop.md`](loop.md).

## 1. Data

**D8. Source.** The JPG release named by the task (`kmmuleelab/Lung_Pathology_Image_JPG`, pinned
revision in `conf/data.yaml`). The authors also host the original SVS slides; we read only their
headers, for the true pixel size (D3).

**D9. Metadata.** The task's label column (`Tumor Subtype`) is the target. Only `Age` and `Sex` are
model inputs; `Pathological_Diagnosis` and the other columns of the official spreadsheet are
near-copies of the label and are never used as inputs.

**D4. Missing images.** The release contains 386 of the 408 JPGs. Six patients have no image at
all (3 Acinar, 2 Lepidic, 1 Papillary) and are excluded (`artifacts/cohort_excluded.csv`):
**203 patients** remain. We keep one image source for every patient rather than mixing in SVS
files for six.

**D5. One slide per patient.** The lowest WSI number (numeric, so WSI-4 precedes WSI-14) among
the patient's slides that have an image. **D2.** All 386 slides were downloaded; the
one-slide-per-patient protocol is the headline throughout.

## 2. Data quirks (TASK.md §2.3)

**D10. Duplicate `SampleNumber` 8377886 → one patient.** All four slides share sex, diagnosis and
label (Papillary); only age differs (68 vs 69). The two errors are not symmetric: treating one
person as two can place the same patient in training and test (leakage); treating two people as
one removes one Papillary patient (< 0.5% of the cohort). Represented by WSI-35.

**D11. Conflicting labels → each slide keeps its own label; the selected slide's label is the
target.** The five patients (8225322, 8240634, 8245975, 8248415, 8248805) show different tissue
blocks with different predominant patterns. Using the label of the slide the model sees (chosen
by D5, before its label is read) keeps image and label consistent and excludes nobody.
Sensitivity analysis: test metrics are also reported without these patients (`results.md`).

Classes (203 patients): In situ 45, Papillary 39, Acinar 30, Solid 29, Micropapillary 26,
Lepidic 19, Cribriform 15.

## 3. Splits

**D12.** Per seed *s* ∈ {0, …, 4}: a test set of 20% of patients (41), stratified by subtype
(`train_test_split(stratify=label, random_state=s)`); the other 162 form the development set,
split into 8 stratified folds (`StratifiedKFold(shuffle=True, random_state=s)`). Each fold
validates once (20–21 patients) while seven train (141–142), matching the task's 70/10/20. The
unit is the patient (`SampleNumber`), so no patient spans two splits. Split files
(`artifacts/splits/seed_<s>.csv`) are written once, checked (disjoint, every class in every test
set and fold) and committed; the stage refuses to overwrite them. **Seed 0 is the headline
split; the agent used seed 0 development data only.**

## 4. Slides and tiles

**D3. Scale.** All originals are 0.1038 µm/px (80×), but each JPG was shrunk to a 65,000-px long
side, so the JPGs range from 0.16 to 0.64 µm/px. A fixed 4× shrink would give 0.65–2.5 µm/px.
We compute each JPG's true pixel size from the SVS header and resample every slide to exactly
**0.5 µm/px (20×)**, the scale the foundation models expect (`artifacts/slide_scale.csv`).

**D13. Resampling.** One anti-aliased Lanczos-3 resize per slide (libvips), saved as a tiled
pyramidal TIFF; no stain normalisation.

**D15. Tissue and tiles.** On the 8 µm/px level: HSV saturation, median filter, per-slide Otsu
threshold (floor 15/255), closing, removal of pieces under 0.02 mm²; pen ink (hue 20–180,
saturation > 40) is removed from the mask. Tiles of **224 × 224 px at 0.5 µm/px (112 µm)** on a
regular grid, **stride 224** (no overlap), kept if ≥ 50% tissue and ≤ 5% ink. Result: 2,040,920
tiles, median 9,560 per slide. All 203 tissue masks were checked visually
(`scripts/06_tile_slides.py` writes QC images).

**E1. Stride check** ([`experiments/E1_tile_stride.md`](experiments/E1_tile_stride.md)):
overlapping tiles (stride 112) change slide vectors by < 2% of between-slide differences
(cosine ≥ 0.99998), below tile-sampling noise, at 4× the cost.

## 5. Foundation models and embeddings

**D19. The three models.**

| Model | Source (pinned) | Tile embedding | Dim |
| --- | --- | --- | --- |
| UNI2-h | `MahmoodLab/UNI2-h` | class token | 1,536 |
| Virchow2 | `paige-ai/Virchow2` | class token + mean patch token | 2,560 |
| GenBio-PathFM | `genbio-ai/genbio-pathfm` | class token per RGB channel, concatenated | 4,608 |

UNI2-h ranks first and Virchow2 and GenBio-PathFM tie second on the THUNDER leaderboard; the three
come from different labs, data and objectives. Prism2 was not used: access was not granted, and
it reuses Virchow2's tile encoder (the least independent choice). GenBio-PathFM requires
`trust_remote_code`; its pinned code was reviewed (plain PyTorch, no network or file access).

**D6. Per-model preprocessing.** Each transform is built from the model's own published
configuration (UNI2-h and Virchow2: ImageNet mean/std, bilinear and bicubic resize;
GenBio-PathFM: its own mean/std). Tiles are already 224 px at 0.5 µm/px. `tests/test_models.py`
fails if a model falls back to a generic transform.

**D16, D7. Storage.** One HDF5 file per slide and model (`artifacts/embeddings/<model>/`), with
tile coordinates, every tile embedding (float32), and model revision, precision, GPU, code
version and a tile-list checksum. Virchow2's class token and mean patch token are stored
separately. Embeddings are computed once and never recomputed inside any search.

**Cost.** UNI2-h and Virchow2: 5.8 GPU-hours each (≈ 98 tiles/s); GenBio-PathFM: 19.9 GPU-hours
(three ViT passes per tile). Disk: ≈ 67 GB for all tile embeddings.

## 6. Tile-to-slide aggregation

**D17.** Two aggregations, both chosen on development data only:
- **Mean pooling** of all tile embeddings: the slide vectors of the baseline rows 1–5
  (`artifacts/features/<model>.npz`).
- **Attention-based multiple-instance learning (ABMIL)** over a fixed random subset of 1,024
  tiles per slide (seeded by the slide number; the same tiles for every model;
  `artifacts/features/tiles/`): the representation of the fused designs, learned inside each
  training fold. The agent selected it (v04): +0.055 macro AUROC over mean pooling on
  development data (0.799 → 0.854). Tiles are never given their slide's label.

## 7. Baselines (rows 1–5 of TASK.md §5.3)

**D21.** One classifier for every row: standardisation and class-weighted multinomial logistic
regression; C from {0.0001, 0.001, 0.01, 0.1, 1} by inner 5-fold CV on the training folds
(macro AUROC). Rows: age and sex; each model alone (mean-pooled vector); the best single model
(chosen on development AUROC, per seed) plus age and sex by plain concatenation (row 5).
Out-of-fold over the 8 folds, seeds 0–4 (`artifacts/runs/`). For the final evaluation every row
is also computed under the fused design's own tile-level recipe, so the comparison holds the
classifier fixed (TASK.md line 162; `results.md`).

Development baselines (mean ± sd over seeds 0–4, macro AUROC): age and sex 0.612 ± 0.016;
UNI2-h 0.787 ± 0.020; Virchow2 0.778 ± 0.020; GenBio-PathFM 0.779 ± 0.013; best single + age and
sex 0.792 ± 0.017.

## 8. Compute and reproducibility

**D18. Compute.** fp32 with TF32 off throughout. Embeddings were computed on NVIDIA RTX PRO 6000
Blackwell GPUs, all model training on NVIDIA GeForce RTX 4090 GPUs; scikit-learn runs on CPU
with one BLAS thread per fit. **Other GPU types run the same code, but their numbers may differ in
the last digits.** One GPU type per stage avoids a hidden confound: E2 (`scripts/experiments/
precision_check.py`) found tile embeddings differ slightly between fp16 and fp32 (cosine ≥ 0.99987)
and between GPU types (≥ 0.99997).

**D14. Reproducibility.** Dataset and model revisions are pinned (`conf/data.yaml`,
`conf/models.yaml`); dependencies are locked (`uv.lock`); the cohort and splits are rebuilt from
committed inputs and checked byte for byte by the tests; every network is trained with fixed
seeds and deterministic algorithms (the final evaluation reproduces its own numbers exactly).
**D1.** Python 3.12, PyTorch 2.8 (CUDA 12.8), transformers 4.x.

## 9. The agent

**D20.** The fusion-search agent (inner layer): [`agent.md`](agent.md).
**D22.** The version loop (outer layer) that improved it over 11 versions: [`loop.md`](loop.md);
its records are in [`../decisions/`](../decisions/README.md).
