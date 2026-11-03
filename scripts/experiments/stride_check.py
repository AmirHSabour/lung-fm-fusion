"""Does overlapping tiling (stride 112) change the slide embedding versus stride 224?

Label-free check on development slides of seed 0 (docs/methods.md D15). The stride-112 grid
contains every stride-224 tile, so each slide is embedded once at stride 112 and the
stride-224 result is the subset of tiles whose corner is a multiple of 224.

For each slide and embedding we compare the mean-pooled slide vectors and report:
  cosine:      cosine similarity between the stride-224 and stride-112 slide vectors;
  d_stride:    their Euclidean distance;
  d_halfsplit: distance between the means of two random halves of the stride-224 tiles
               (how much the slide vector moves from tile sampling alone);
  d_between:   median distance between the stride-112 vectors of different slides.
If d_stride is small next to d_between and comparable to d_halfsplit, overlap adds no
information that mean pooling can use.

Usage (from the repo root; one model per GPU):
    uv run python -m scripts.experiments.stride_check --model uni2h --device cuda:0
    uv run python -m scripts.experiments.stride_check --model virchow2 --device cuda:1
    uv run python -m scripts.experiments.stride_check --analyze
"""
from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch

from src.data.tiling import tile_grid
from src.data.tissue import segment_tissue
from src.embeddings.extract import embed_slide
from src.embeddings.models import load_uni2h, load_virchow2, model_transform
from src.utils.config import load_config

SLIDES = ["WSI-211", "WSI-140", "WSI-288", "WSI-97", "WSI-238"]  # seed-0 development slides, small to large
OUT = Path("artifacts/experiments/stride_check")
LOADERS = {"uni2h": load_uni2h, "virchow2": load_virchow2}


def run(key: str, device: str) -> None:
    pre, models = load_config("conf/preprocess.yaml"), load_config("conf/models.yaml")
    tc, tl = pre["tissue"], pre["tiling"]
    torch.cuda.set_device(device)
    model = LOADERS[key](Path(models["root"]) / models["models"][key]["local_dir"]).to(device)
    transform = model_transform(model)
    rng = np.random.default_rng(0)
    rows, t_all = [], time.time()
    for wsi in SLIDES:
        slide = Path(pre["out_dir"]) / f"{wsi}.tif"
        im = pyvips.Image.new_from_file(str(slide))
        tm = segment_tissue(slide, pre["resample"]["target_mpp"], mask_mpp=tc["mask_mpp"],
                            median_um=tc["median_um"], min_saturation=tc["min_saturation"],
                            closing_um=tc["closing_um"], min_component_mm2=tc["min_component_mm2"])
        tiles = tile_grid(tm.mask, im.width, im.height, tile_px=tl["tile_px"],
                          stride_px=tl["tile_px"] // 2, min_tissue=tl["min_tissue"])
        on_224 = ((tiles.x % tl["tile_px"] == 0) & (tiles.y % tl["tile_px"] == 0)).to_numpy()
        t0 = time.time()
        feats = embed_slide(model, key, transform, slide, tiles[["x", "y"]].to_numpy(),
                            tile_px=tl["tile_px"], device=device)
        secs = time.time() - t0
        for name, f in feats.items():
            half = rng.permutation(np.flatnonzero(on_224))
            a, b = half[: len(half) // 2], half[len(half) // 2:]
            rows.append(dict(model=key, feature=name, WSI_ID=wsi, n_224=int(on_224.sum()), n_112=len(f),
                             mean_224=f[on_224].mean(0), mean_112=f.mean(0),
                             d_halfsplit=float(np.linalg.norm(f[a].mean(0) - f[b].mean(0)))))
        print(f"{key} {wsi}: {on_224.sum()} tiles at stride 224, {len(tiles)} at stride 112, "
              f"{secs:.0f}s ({len(tiles) / secs:.0f} tiles/s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(pd.DataFrame(rows), OUT / f"{key}.pkl")
    print(f"{key}: done in {(time.time() - t_all) / 60:.1f} min")


def analyze() -> None:
    df = pd.concat(pd.read_pickle(p) for p in sorted(OUT.glob("*.pkl")))
    out = []
    for (key, name), g in df.groupby(["model", "feature"], sort=False):
        m112 = dict(zip(g.WSI_ID, g.mean_112))
        d_between = np.median([np.linalg.norm(m112[i] - m112[j]) for i, j in combinations(m112, 2)])
        for r in g.itertuples():
            out.append(dict(model=key, feature=name, WSI_ID=r.WSI_ID, n_224=r.n_224, n_112=r.n_112,
                            cosine=float(r.mean_224 @ r.mean_112 /
                                         (np.linalg.norm(r.mean_224) * np.linalg.norm(r.mean_112))),
                            d_stride=float(np.linalg.norm(r.mean_224 - r.mean_112)),
                            d_halfsplit=r.d_halfsplit, d_between=float(d_between)))
    res = pd.DataFrame(out)
    res["stride_vs_between"] = res.d_stride / res.d_between
    res["stride_vs_halfsplit"] = res.d_stride / res.d_halfsplit
    res.round(5).to_csv(OUT.with_suffix(".csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(f"-> {OUT.with_suffix('.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(LOADERS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze() if args.analyze else run(args.model, args.device)


if __name__ == "__main__":
    main()
