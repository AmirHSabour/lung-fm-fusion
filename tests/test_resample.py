"""Resampling produces the expected size and records the target pixel size."""
from __future__ import annotations

import numpy as np
import pytest
import pyvips

from src.data.resample import expected_size, is_done, resample_slide


@pytest.mark.parametrize("src_mpp", [0.163, 0.305, 0.635])  # finest, typical, coarsest JPG
def test_resample_slide(tmp_path, src_mpp):
    rng = np.random.default_rng(0)
    src = tmp_path / "slide.jpg"
    pyvips.Image.new_from_array(rng.integers(0, 256, (1000, 1500, 3), dtype=np.uint8)).jpegsave(str(src))
    dst = tmp_path / "slide.tif"

    info = resample_slide(src, dst, src_mpp, 0.5)

    w, h = expected_size(1500, 1000, src_mpp, 0.5)
    out = pyvips.Image.new_from_file(str(dst))
    assert (out.width, out.height, out.bands) == (w, h, 3)
    assert is_done(dst, w, h)
    assert 1000 / out.xres == pytest.approx(0.5) and 1000 / out.yres == pytest.approx(0.5)
    assert abs(info["out_mpp_x"] - 0.5) < 0.5 * 1 / w + 1e-9  # off only by output-size rounding
    assert not dst.with_name(dst.name + ".partial").exists()
