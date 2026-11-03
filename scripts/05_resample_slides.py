"""Stage 05: resample every cohort slide to 0.5 microns per pixel (pyramidal TIFF).

Idempotent: slides whose output already exists with the expected size are skipped,
and outputs are written atomically, so an interrupted run can simply be restarted.

Usage (from the repo root):
    uv run python -m scripts.05_resample_slides [--config conf/preprocess.yaml] [--all-slides]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.data.resample import expected_size, is_done, resample_slide
from src.utils.config import load_config


def _run(job: dict, rs: dict) -> dict:
    t = time.time()
    info = resample_slide(job["src"], job["dst"], job["jpg_mpp"], rs["target_mpp"], kernel=rs["kernel"],
                          tile_size=rs["tile_size"], jpeg_quality=rs["jpeg_quality"])
    return dict(WSI_ID=job["WSI_ID"], **info, seconds=round(time.time() - t, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/preprocess.yaml"))
    ap.add_argument("--all-slides", action="store_true",
                    help="every slide with a JPG (386), not just one per patient (203)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rs, images, out_dir = cfg["resample"], Path(cfg["images_dir"]), Path(cfg["out_dir"])

    scale = pd.read_csv(cfg["slide_scale_csv"]).dropna(subset=["jpg_width"])
    if not args.all_slides:
        scale = scale[scale.WSI_ID.isin(pd.read_csv(cfg["cohort_csv"]).WSI_ID)]
    jobs, skipped = [], 0
    for r in scale.itertuples():
        dst = out_dir / f"{r.WSI_ID}.tif"
        w, h = expected_size(int(r.jpg_width), int(r.jpg_height), r.jpg_mpp, rs["target_mpp"])
        if is_done(dst, w, h):
            skipped += 1
        else:
            jobs.append(dict(WSI_ID=r.WSI_ID, src=images / f"{r.WSI_ID}.jpg", dst=dst, jpg_mpp=r.jpg_mpp))
    print(f"{len(scale)} slides | {skipped} already done | {len(jobs)} to resample", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(rs["workers"]) as pool:
        futures = {pool.submit(_run, j, rs): j["WSI_ID"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['WSI_ID']}: {row['out_width']}x{row['out_height']} "
                  f"@ {row['out_mpp_x']:.4f} um/px, {row['out_bytes'] / 1e9:.2f} GB, {row['seconds']}s "
                  f"| elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    manifest = Path(cfg["manifest_csv"])
    if rows:
        old = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
        new = pd.DataFrame(rows)
        old = old[~old.WSI_ID.isin(new.WSI_ID)] if len(old) else old
        pd.concat([old, new]).sort_values("WSI_ID").to_csv(manifest, index=False)
        print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
