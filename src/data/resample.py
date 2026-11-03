"""Resample a CLWD JPG to a fixed microns-per-pixel and save it as a pyramidal TIFF.

The Hugging Face JPGs each have their own pixel size (0.16 to 0.64 microns per pixel,
measured in artifacts/slide_scale.csv). Every slide is resampled once to the target
(0.5, i.e. 20x) with an anti-aliased Lanczos kernel, then stored as a tiled, pyramidal,
JPEG-compressed BigTIFF whose resolution tags record the new pixel size. Later steps
(tissue detection, tiling, embedding) read these files and never see the original scale.
"""
from __future__ import annotations

from pathlib import Path

import pyvips


def expected_size(width: int, height: int, src_mpp: float, target_mpp: float) -> tuple[int, int]:
    """Output size in pixels, as computed by libvips' resize (round to nearest)."""
    scale = src_mpp / target_mpp
    return round(width * scale), round(height * scale)


def is_done(dst: Path, width: int, height: int) -> bool:
    """True if dst exists and has the expected full-resolution size."""
    if not dst.is_file():
        return False
    try:
        im = pyvips.Image.new_from_file(str(dst))
    except pyvips.Error:
        return False
    return (im.width, im.height) == (width, height)


def resample_slide(src: Path, dst: Path, src_mpp: float, target_mpp: float, *,
                   kernel: str = "lanczos3", tile_size: int = 512, jpeg_quality: int = 90) -> dict:
    """Resample src to target_mpp and write a pyramidal TIFF to dst (atomically).

    Returns the source and output sizes and the achieved microns per pixel along each axis,
    which differs from target_mpp only by the rounding of the output size.
    """
    im = pyvips.Image.new_from_file(str(src), access="sequential")
    if im.bands != 3 or im.format != "uchar":
        raise ValueError(f"{src}: expected 8-bit RGB, got {im.bands} bands of {im.format}")
    out = im.resize(src_mpp / target_mpp, kernel=kernel)

    px_per_mm = 1000.0 / target_mpp  # libvips resolution unit is pixels per millimetre
    tmp = dst.with_name(dst.name + ".partial")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.tiffsave(str(tmp), tile=True, tile_width=tile_size, tile_height=tile_size, pyramid=True,
                 compression="jpeg", Q=jpeg_quality, bigtiff=True,
                 xres=px_per_mm, yres=px_per_mm, resunit="cm")
    tmp.replace(dst)
    return dict(src_width=im.width, src_height=im.height, src_mpp=src_mpp,
                out_width=out.width, out_height=out.height,
                out_mpp_x=src_mpp * im.width / out.width, out_mpp_y=src_mpp * im.height / out.height,
                out_bytes=dst.stat().st_size)
