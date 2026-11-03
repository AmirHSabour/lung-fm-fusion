"""Regular tile grid over a slide, keeping tiles that are mostly tissue.

Tiles are tile_px x tile_px at the slide's own resolution (0.5 microns per pixel after
resampling). The grid starts at (0, 0) and steps by stride_px, so it is deterministic,
and a grid with half the stride contains every tile of the coarser grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _box_fraction(m: np.ndarray, x0, x1, y0, y1) -> np.ndarray:
    """Share of True pixels of m in each box [x0, x1) x [y0, y1), via a summed-area table."""
    s = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.int64)
    s[1:, 1:] = m.astype(np.int64).cumsum(0).cumsum(1)
    return (s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]) / np.maximum((x1 - x0) * (y1 - y0), 1)


def tile_grid(mask: np.ndarray, slide_w: int, slide_h: int, *, tile_px: int = 224,
              stride_px: int = 224, min_tissue: float = 0.5, ink: np.ndarray | None = None,
              max_ink: float = 0.05) -> pd.DataFrame:
    """Top-left corners (x, y), tissue and ink fractions of every kept tile, in slide pixels.

    A tile is kept if at least min_tissue of its footprint on the mask is tissue and at
    most max_ink of it is pen ink. Fractions are exact shares of mask pixels.
    """
    mh, mw = mask.shape
    sx, sy = mw / slide_w, mh / slide_h

    xs = np.arange(0, slide_w - tile_px + 1, stride_px)
    ys = np.arange(0, slide_h - tile_px + 1, stride_px)
    gx, gy = np.meshgrid(xs, ys)
    x0 = np.clip(np.round(gx * sx).astype(int), 0, mw)
    x1 = np.clip(np.round((gx + tile_px) * sx).astype(int), 0, mw)
    y0 = np.clip(np.round(gy * sy).astype(int), 0, mh)
    y1 = np.clip(np.round((gy + tile_px) * sy).astype(int), 0, mh)
    frac = _box_fraction(mask, x0, x1, y0, y1)
    ink_frac = _box_fraction(ink, x0, x1, y0, y1) if ink is not None else np.zeros_like(frac)

    keep = (frac >= min_tissue) & (ink_frac <= max_ink)
    return pd.DataFrame({"x": gx[keep], "y": gy[keep], "tissue_frac": frac[keep].round(4),
                         "ink_frac": ink_frac[keep].round(4)})


def qc_image(preview: np.ndarray, mask: np.ndarray, tiles: pd.DataFrame, slide_w: int,
             tile_px: int, title: str = "") -> Image.Image:
    """Side by side: the slide, its tissue mask, and the kept tiles outlined in green."""
    h, w = mask.shape
    s = w / slide_w
    base = Image.fromarray(preview[:h, :w])
    mask_img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("RGB")
    over = base.copy()
    draw = ImageDraw.Draw(over, "RGBA")
    for x, y in zip(tiles.x, tiles.y):
        draw.rectangle([x * s, y * s, (x + tile_px) * s - 1, (y + tile_px) * s - 1],
                       fill=(0, 200, 0, 70), outline=(0, 120, 0, 160))
    sheet = Image.new("RGB", (3 * w + 20, h + 24), "white")
    for i, im in enumerate((base, mask_img, over)):
        sheet.paste(im, (i * (w + 10), 24))
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(0, 0, 0))
    return sheet
