"""Are tile embeddings numerically consistent across precision and GPU type? (experiment E2)

Embeds the same 2,048 tiles of one development slide (WSI-140) under several conditions
and compares each against a reference: fp16 autocast vs fp32 on one GPU, and fp16 on two
different GPU architectures. Reports, per embedding, the cosine similarity between tile
vectors (minimum and median) and between mean-pooled slide vectors. No labels.

Usage (from the repo root):
    uv run python -m scripts.experiments.precision_check --device cuda:2                  # fp16 vs fp32
    uv run python -m scripts.experiments.precision_check --compare-devices cuda:0 cuda:2  # GPU types
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.embeddings.extract import embed_slide
from src.embeddings.models import load_uni2h, load_virchow2, model_transform
from src.utils.config import load_config

SLIDE, N_TILES = "WSI-140", 2048
OUT_DIR = Path("artifacts/experiments")


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def compare(out_a: dict, out_b: dict, label: str, key: str, n: int) -> list[dict]:
    rows = []
    for name in out_a:
        a, b = out_a[name], out_b[name]
        c = cosine_rows(a, b)
        ma, mb = a.mean(0), b.mean(0)
        rows.append(dict(comparison=label, model=key, feature=name, n_tiles=n, tile_cos_min=c.min(),
                         tile_cos_median=np.median(c),
                         slide_cos=float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb)))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0", help="GPU for the fp16 vs fp32 comparison")
    ap.add_argument("--compare-devices", nargs=2, metavar=("A", "B"),
                    help="compare fp16 embeddings on two GPUs instead")
    args = ap.parse_args()
    cfg, models = load_config("conf/embeddings.yaml"), load_config("conf/models.yaml")
    xy = pd.read_csv(Path(cfg["tiles_dir"]) / f"{SLIDE}.csv")[["x", "y"]].to_numpy()[:N_TILES]
    slide = Path(cfg["slides_dir"]) / f"{SLIDE}.tif"
    rows = []
    for key, loader in (("uni2h", load_uni2h), ("virchow2", load_virchow2)):
        model_dir = Path(models["root"]) / models["models"][key]["local_dir"]
        if args.compare_devices:
            outs, names = [], []
            for dev in args.compare_devices:
                model = loader(model_dir).to(dev)
                outs.append(embed_slide(model, key, model_transform(model), slide, xy,
                                        workers=cfg["workers"], device=dev, precision="fp16"))
                names.append(torch.cuda.get_device_name(dev))
                del model
            rows += compare(*outs, f"fp16: {names[0]} vs {names[1]}", key, len(xy))
        else:
            model = loader(model_dir).to(args.device)
            tf = model_transform(model)
            out = {p: embed_slide(model, key, tf, slide, xy, workers=cfg["workers"], device=args.device,
                                  precision=p) for p in ("fp16", "fp32")}
            rows += compare(out["fp16"], out["fp32"], f"fp16 vs fp32 on {torch.cuda.get_device_name(args.device)}",
                            key, len(xy))
            del model
    df = pd.DataFrame(rows)
    out = OUT_DIR / ("device_check.csv" if args.compare_devices else "precision_check.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
