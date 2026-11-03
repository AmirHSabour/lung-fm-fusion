"""Stage 06: tissue segmentation and tiling of every resampled cohort slide.

For each slide: tissue and pen-ink masks at about 8 microns per pixel, a 224 px grid at 0.5 microns
per pixel, and the tiles that are at least half tissue and at most 5% ink. Writes the kept tile positions
(artifacts/tiles/<WSI_ID>.csv), one QC picture per slide and a per-slide summary.
Uses no labels.

Usage (from the repo root):
    uv run python -m scripts.06_tile_slides [--slides WSI-140 WSI-97 ...]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyvips

from src.data.tiling import qc_image, tile_grid
from src.data.tissue import segment_tissue
from src.utils.config import load_config


def _run(wsi: str, cfg: dict) -> dict:
    t0 = time.time()
    tc, tl, mpp = cfg["tissue"], cfg["tiling"], cfg["resample"]["target_mpp"]
    slide = Path(cfg["out_dir"]) / f"{wsi}.tif"
    w, h = (lambda im: (im.width, im.height))(pyvips.Image.new_from_file(str(slide)))
    tm = segment_tissue(slide, mpp, mask_mpp=tc["mask_mpp"], median_um=tc["median_um"],
                        min_saturation=tc["min_saturation"], closing_um=tc["closing_um"],
                        min_component_mm2=tc["min_component_mm2"], ink_hue=tuple(tc["ink_hue"]),
                        ink_min_saturation=tc["ink_min_saturation"], ink_min_mm2=tc["ink_min_mm2"],
                        ink_dilate_um=tc["ink_dilate_um"])
    tiles = tile_grid(tm.mask, w, h, tile_px=tl["tile_px"], stride_px=tl["stride_px"],
                      min_tissue=tl["min_tissue"], ink=tm.ink, max_ink=tl["max_ink"])
    tiles.to_csv(Path(tl["tiles_dir"]) / f"{wsi}.csv", index=False)
    tissue_mm2 = tm.mask.sum() * tm.mask_mpp ** 2 / 1e6
    qc_image(tm.preview, tm.mask, tiles, w, tl["tile_px"],
             title=f"{wsi}  tissue {tissue_mm2:.0f} mm2  ink {tm.ink.mean():.1%}  threshold {tm.threshold}  "
                   f"tiles {len(tiles)}"
             ).save(Path(tl["qc_dir"]) / f"{wsi}.jpg", quality=85)
    return dict(WSI_ID=wsi, width=w, height=h, mask_mpp=round(tm.mask_mpp, 3),
                sat_threshold=tm.threshold, tissue_mm2=round(tissue_mm2, 2),
                ink_area_frac=round(float(tm.ink.mean()), 5), n_tiles=len(tiles),
                seconds=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/preprocess.yaml"))
    ap.add_argument("--slides", nargs="+", help="WSI_IDs to process (default: whole cohort)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tl = cfg["tiling"]
    for d in (tl["tiles_dir"], tl["qc_dir"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    slides = args.slides or pd.read_csv(cfg["cohort_csv"])["WSI_ID"].tolist()
    t0 = time.time()
    with ProcessPoolExecutor(tl["workers"]) as pool:
        rows = list(pool.map(_run, slides, [cfg] * len(slides)))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False) if len(summary) <= 20 else summary.describe().round(1).to_string())
    print(f"{len(summary)} slides, {summary.n_tiles.sum()} tiles, {(time.time() - t0) / 60:.1f} min")
    if args.slides is None:
        summary.to_csv(tl["summary_csv"], index=False)
        print(f"summary -> {tl['summary_csv']}")


if __name__ == "__main__":
    main()
