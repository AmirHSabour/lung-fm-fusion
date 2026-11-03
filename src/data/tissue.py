"""Tissue segmentation on a low-resolution level of a resampled slide.

The method follows the CLWD authors' preprocessing and the CLAM default: smooth the
image, convert to HSV, threshold the saturation channel (stained tissue is saturated;
glass, dust and shadows are not), then clean the mask. The threshold is chosen per
slide with Otsu's method, with a floor so that a slide with very little tissue cannot
split the background noise into two classes.

Pen ink is also saturated, so it is detected separately by hue: H&E eosin and
hematoxylin sit at hues of about 185-255 (libvips 0-255 scale), while blue and green
marker ink sit at about 20-180 (measured on this dataset: under 0.06% of stained pixels
on clean slides fall in that range, versus 33% on pen-marked slides).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvips
from scipy import ndimage


@dataclass(frozen=True)
class TissueMask:
    mask: np.ndarray        # bool (H, W) at mask_mpp
    preview: np.ndarray     # uint8 RGB (H, W, 3) of the same level, for QC
    ink: np.ndarray         # bool (H, W): pen ink (dilated), never counted as tissue
    mask_mpp: float         # microns per mask pixel (per axis)
    threshold: int          # saturation threshold actually used (0-255)


def otsu_threshold(values: np.ndarray) -> int:
    """Otsu's threshold for 8-bit values: maximises the between-class variance."""
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(between))


def pick_level(slide: Path, slide_mpp: float, target_mpp: float) -> tuple[int, float, float]:
    """Pyramid level whose pixel size is closest to target_mpp; returns (page, mpp_x, mpp_y)."""
    base = pyvips.Image.new_from_file(str(slide))
    best = None
    for page in range(base.get("n-pages")):
        lvl = pyvips.Image.new_from_file(str(slide), page=page)
        mpp_x, mpp_y = slide_mpp * base.width / lvl.width, slide_mpp * base.height / lvl.height
        if best is None or abs(np.log(mpp_x / target_mpp)) < abs(np.log(best[1] / target_mpp)):
            best = (page, mpp_x, mpp_y)
    return best


def segment_tissue(slide: Path, slide_mpp: float, *, mask_mpp: float = 8.0, median_um: float = 56.0,
                   min_saturation: int = 15, min_component_mm2: float = 0.02, closing_um: float = 16.0,
                   ink_hue: tuple[int, int] = (20, 180), ink_min_saturation: int = 40,
                   ink_min_mm2: float = 0.02, ink_dilate_um: float = 16.0) -> TissueMask:
    """Tissue and pen-ink masks of a pyramidal slide at about mask_mpp microns per pixel."""
    page, mpp_x, mpp_y = pick_level(slide, slide_mpp, mask_mpp)
    rgb = pyvips.Image.new_from_file(str(slide), page=page)
    preview = rgb.numpy()
    hsv = rgb.colourspace("hsv").numpy()
    hue, sat = hsv[..., 0], hsv[..., 1]

    ink = (sat > ink_min_saturation) & (hue >= ink_hue[0]) & (hue < ink_hue[1])
    labels, n = ndimage.label(ink)  # pen strokes are large connected marks; isolated pixels are not
    if n:
        areas = ndimage.sum(ink, labels, index=np.arange(1, n + 1)) * mpp_x * mpp_y / 1e6  # mm^2
        ink = np.isin(labels, 1 + np.flatnonzero(areas >= ink_min_mm2))
    d = max(1, int(round(ink_dilate_um / mpp_x)))
    ink = ndimage.binary_dilation(ink, structure=np.ones((2 * d + 1, 2 * d + 1), bool))

    k = max(3, int(round(median_um / mpp_x)) | 1)          # odd median kernel
    sat = ndimage.median_filter(sat, size=k)
    threshold = max(otsu_threshold(sat), min_saturation)
    mask = sat > threshold

    r = max(1, int(round(closing_um / mpp_x)))
    mask = ndimage.binary_closing(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    labels, n = ndimage.label(mask)
    if n:
        areas = ndimage.sum(mask, labels, index=np.arange(1, n + 1)) * mpp_x * mpp_y / 1e6  # mm^2
        mask = np.isin(labels, 1 + np.flatnonzero(areas >= min_component_mm2))
    mask &= ~ink
    return TissueMask(mask=mask, preview=preview, ink=ink, mask_mpp=(mpp_x + mpp_y) / 2, threshold=threshold)
