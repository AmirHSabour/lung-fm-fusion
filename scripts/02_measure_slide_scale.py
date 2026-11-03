"""Stage 02: measure every JPG's true microns per pixel (artifacts/slide_scale.csv).

One row per WSI_ID in the metadata. Reads only file headers: the original SVS headers
over HTTP byte ranges and the local JPG headers. The output is committed, so later
stages do not depend on the remote server.

Usage (from the repo root):
    uv run python -m scripts.02_measure_slide_scale [--workers 16]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.data.slide_scale import jpg_header, svs_header
from src.utils.config import load_config

TARGET_MPP = 0.5  # only used for the convenience column resize_to_0p5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/data.yaml"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, help="output CSV (default: slide_scale_csv from the config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    images, base_url = Path(cfg["images_dir"]), cfg["dataset"]["svs_url"]
    out = args.out or Path(cfg["slide_scale_csv"])

    ids = pd.read_csv(cfg["metadata_csv"])["WSI_ID"].tolist()

    def row(wsi: str) -> dict:
        return dict(WSI_ID=wsi, **svs_header(base_url, f"{wsi}.svs"), **jpg_header(images / f"{wsi}.jpg"))

    with ThreadPoolExecutor(args.workers) as pool:
        df = pd.DataFrame(list(pool.map(row, ids)))
    if "svs_error" in df and df["svs_error"].notna().any():
        raise SystemExit(f"could not read SVS headers: {df.loc[df.svs_error.notna(), 'WSI_ID'].tolist()}")
    df["downsample"] = df["svs_width"] / df["jpg_width"]
    df["downsample_h"] = df["svs_height"] / df["jpg_height"]
    df["jpg_mpp"] = df["svs_mpp"] * df["downsample"]
    df["resize_to_0p5"] = df["jpg_mpp"] / TARGET_MPP  # multiply JPG dimensions by this to reach 0.5
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} slides, {df.jpg_width.notna().sum()} with a JPG, "
          f"JPG microns per pixel {df.jpg_mpp.min():.3f}-{df.jpg_mpp.max():.3f}")


if __name__ == "__main__":
    main()
