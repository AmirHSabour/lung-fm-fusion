# E1. Tile stride: does overlapping tiling change the slide representation?

**Decision supported:** D15 (tiling stride). **Status:** complete; stride 224 px fixed.

## Question

Each slide is cut into 224 × 224 px tiles at 0.5 µm/px (112 × 112 µm). The stride is the
step between neighbouring tiles. A stride of 224 px gives non-overlapping tiles; a stride
of 112 px gives 50% overlap and four times as many tiles. We asked whether overlap yields
a different, potentially more informative slide representation.

## Rationale for testing

Non-overlapping tiling is the convention of the CLWD authors' own pipeline and of the
UNI, Virchow and Prism pipelines. Overlap could in principle help by showing structures
that a tile boundary would otherwise cut, but it also multiplies extraction cost by four.
Because the encoders are frozen and tile embeddings are pooled into one slide vector, we
expected overlap to add redundant, highly correlated tiles rather than new information.
We tested this expectation directly instead of assuming it.

## Design

- **Slides:** five development slides of seed 0 (WSI-211, WSI-140, WSI-288, WSI-97,
  WSI-238), spanning small to large tissue areas. No test slides, no labels.
- **Encoders:** UNI2-h (class token, 1,536-d) and Virchow2 (class token and mean of patch
  tokens, 1,280-d each), with their published preprocessing.
- **Procedure:** each slide was embedded once on the stride-112 grid. That grid contains
  every stride-224 tile, so the stride-224 representation is the subset of tiles whose
  corner lies on multiples of 224. Slide vectors were obtained by mean pooling.
- **Tiles per slide:** 3,341–17,936 at stride 224; 13,344–71,733 at stride 112.

## Measures

For each slide and embedding:

- **Cosine similarity** between the stride-224 and stride-112 slide vectors.
- **Stride effect** (*d*<sub>stride</sub>): Euclidean distance between the two vectors.
- **Sampling noise** (*d*<sub>half</sub>): distance between the mean vectors of two
  random halves of the stride-224 tiles, i.e. how much a slide vector moves from tile
  sampling alone.
- **Between-slide signal** (*d*<sub>between</sub>): median distance between the
  slide vectors of different slides, i.e. the variation a classifier relies on.

Overlap is irrelevant if *d*<sub>stride</sub> is negligible relative to
*d*<sub>between</sub> and no larger than *d*<sub>half</sub>.

## Results

| Embedding | Min. cosine | *d*<sub>stride</sub> / *d*<sub>between</sub>, median (max) | *d*<sub>stride</sub> / *d*<sub>half</sub>, median (max) |
| --- | --- | --- | --- |
| UNI2-h class token | 0.99998 | 0.5% (0.9%) | 21% (27%) |
| Virchow2 class token | 0.99999 | 0.4% (0.6%) | 13% (20%) |
| Virchow2 patch mean | 0.99999 | 1.0% (1.6%) | 16% (18%) |

Per-slide values are in `artifacts/experiments/stride_check.csv`.

## Interpretation

Moving from stride 224 to stride 112 changes the slide vector by less than 2% of the
difference between slides, and by less than the change caused by randomly discarding
half of the tiles. Overlapping tiles therefore carry no information that mean pooling
can use. They would, however, quadruple extraction time and storage, and change the
tile distribution on which Prism2's slide encoder was trained.

## Decision

Stride 224 px (no overlap) is used for all slides.

## Limitations

The test covers mean pooling on five slides. Attention-based pooling weighs individual
tiles and could in principle react differently; given the near-identical tile
populations, a material difference is not expected.

## Reproduction and cost

```bash
uv run python -m scripts.experiments.stride_check --model uni2h --device cuda:0
uv run python -m scripts.experiments.stride_check --model virchow2 --device cuda:1
uv run python -m scripts.experiments.stride_check --analyze
```

About 210,000 tiles per encoder; 14.8 min per encoder on one RTX 4090 (~250 tiles/s).
