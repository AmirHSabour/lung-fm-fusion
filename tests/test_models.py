"""The tile encoders load from local weights with their published preprocessing.

Skipped when the weights are not downloaded or no GPU is available.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.embeddings.models import load_genbio_pathfm, load_uni2h, load_virchow2, model_transform
from src.utils.config import load_config

CFG = load_config("conf/models.yaml")
ROOT = Path(CFG["root"])

CASES = {
    "uni2h": (load_uni2h, (2, 1536), "bilinear"),
    "virchow2": (load_virchow2, (2, 261, 1280), "bicubic"),
}


@pytest.mark.parametrize("key", CASES)
def test_encoder_forward_and_transform(key):
    loader, shape, interp = CASES[key]
    model_dir = ROOT / CFG["models"][key]["local_dir"]
    if not model_dir.exists():
        pytest.skip(f"{model_dir} not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    model = loader(model_dir).cuda()
    cfg = model.pretrained_cfg
    assert tuple(cfg["mean"]) == pytest.approx((0.485, 0.456, 0.406))  # ImageNet, not timm's 0.5 default
    assert tuple(cfg["std"]) == pytest.approx((0.229, 0.224, 0.225))
    assert cfg["interpolation"] == interp and cfg["crop_pct"] == 1.0
    assert "224" in str(model_transform(model))

    x = torch.rand(2, 3, 224, 224, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        y = model(x)
    assert tuple(y.shape) == shape and torch.isfinite(y).all()


def test_genbio_pathfm_forward_and_transform():
    model_dir = ROOT / CFG["models"]["genbio_pathfm"]["local_dir"]
    if not model_dir.exists():
        pytest.skip(f"{model_dir} not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    model = load_genbio_pathfm(model_dir).cuda()
    norm = model_transform(model).transforms[-1]
    assert tuple(norm.mean) == pytest.approx((0.697, 0.575, 0.728))   # model card
    assert tuple(norm.std) == pytest.approx((0.188, 0.240, 0.187))
    assert not model.training                                          # eval: no RoPE coordinate jitter
    with torch.inference_mode():
        y = model(torch.rand(2, 3, 224, 224, device="cuda"))
    assert tuple(y.shape) == (2, 4608) and torch.isfinite(y).all()
