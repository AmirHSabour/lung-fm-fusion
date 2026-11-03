"""Tissue segmentation and tile-grid geometry on synthetic slides."""
from __future__ import annotations

import numpy as np
import pyvips

from src.data.tiling import tile_grid
from src.data.tissue import otsu_threshold, segment_tissue


def test_otsu_separates_two_modes():
    values = np.concatenate([np.full(1000, 3), np.full(500, 150)]).astype(np.uint8)
    assert 3 <= otsu_threshold(values) < 150


def test_tile_grid_fractions_and_stride_nesting():
    mask = np.zeros((100, 100), bool)
    mask[:, :50] = True                         # left half of the slide is tissue
    full = tile_grid(mask, 1600, 1600, tile_px=224, stride_px=224, min_tissue=0.5)
    half = tile_grid(mask, 1600, 1600, tile_px=224, stride_px=112, min_tissue=0.5)
    assert (full.x + 224 <= 800 + 112).all()                       # only tiles that are mostly left-half
    assert set(zip(full.x, full.y)) <= set(zip(half.x, half.y))   # stride-112 grid contains stride 224
    assert (full.x % 224 == 0).all() and (full.y % 224 == 0).all()


def test_ink_excludes_tiles():
    mask = np.ones((100, 100), bool)
    ink = np.zeros((100, 100), bool)
    ink[:, :20] = True
    tiles = tile_grid(mask, 1600, 1600, tile_px=224, stride_px=224, ink=ink, max_ink=0.05)
    assert (tiles.ink_frac <= 0.05).all() and (tiles.x >= 224).all()


def test_segment_tissue_finds_stain_and_ignores_pen(tmp_path):
    rgb = np.full((2048, 2048, 3), 235, np.uint8)       # glass
    rgb[400:1600, 400:1200] = (200, 120, 190)           # pink/purple H&E-like tissue
    rgb[400:1600, 1500:1700] = (40, 60, 200)            # blue pen stroke on glass
    path = tmp_path / "slide.tif"
    pyvips.Image.new_from_array(rgb).tiffsave(str(path), tile=True, pyramid=True, compression="jpeg", Q=95)
    tm = segment_tissue(path, 0.5, mask_mpp=4.0, min_component_mm2=0.001, ink_min_mm2=0.001)
    s = tm.mask.shape[1] / 2048
    tissue_col, pen_col = int(800 * s), int(1600 * s)
    assert tm.mask[int(1000 * s), tissue_col]          # stain is tissue
    assert not tm.mask[int(1000 * s), pen_col]         # pen is not tissue
    assert tm.ink[int(1000 * s), pen_col]              # ...and is flagged as ink
    assert not tm.mask[int(200 * s), int(200 * s)]     # glass is not tissue
