"""Load the tile encoders from local weights with each model's own preprocessing.

Important: ``pretrained_cfg_overlay=dict(file=...)`` alone makes timm fall back to its
generic architecture defaults for preprocessing (mean/std 0.5, crop_pct 0.9). Each
repo's ``config.json`` carries the model's published ``pretrained_cfg`` (ImageNet
mean/std, crop_pct 1.0, bilinear for UNI2-h, bicubic for Virchow2), so it is merged into
the overlay here (D6).
"""
from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoModel, PreTrainedModel

# Keys of the Hub config's ``pretrained_cfg`` that define the model's own preprocessing.
_PREPROC_KEYS = ("input_size", "interpolation", "crop_pct", "crop_mode", "mean", "std")


def local_pretrained_cfg(model_dir: Path, weights: str) -> dict:
    """timm overlay: load weights from a local file and keep the published preprocessing."""
    cfg = json.loads((model_dir / "config.json").read_text())["pretrained_cfg"]
    overlay = {k: cfg[k] for k in _PREPROC_KEYS if k in cfg}
    overlay.update(file=str(model_dir / weights), custom_load=False, num_classes=0)
    return overlay


def load_uni2h(model_dir: Path) -> torch.nn.Module:
    """UNI2-h (architecture arguments from the model card): (B, 3, 224, 224) -> (B, 1536)."""
    return timm.create_model(
        "vit_giant_patch14_224", pretrained=True,
        pretrained_cfg_overlay=local_pretrained_cfg(model_dir, "pytorch_model.bin"),
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
        dynamic_img_size=True,
    ).eval()


def load_virchow2(model_dir: Path) -> torch.nn.Module:
    """Virchow2: (B, 3, 224, 224) -> (B, 261, 1280) tokens.

    Token 0 is the class token, tokens 1-4 are registers, tokens 5: are the 256 patch tokens.
    """
    cfg = json.loads((model_dir / "config.json").read_text())
    return timm.create_model(
        "vit_huge_patch14_224", pretrained=True,
        pretrained_cfg_overlay=local_pretrained_cfg(model_dir, "model.safetensors"),
        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, **cfg["model_args"],
    ).eval()


def load_genbio_pathfm(model_dir: Path) -> torch.nn.Module:
    """GenBio-PathFM (its own model code, pinned and reviewed; docs/methods.md D19).

    (B, 3, 224, 224) -> (B, 4608): each RGB channel is encoded separately as a one-channel
    image and the three class tokens (1536 each) are concatenated, as in the model card.
    """
    return AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                     torch_dtype=torch.float32).eval()


def model_transform(model: torch.nn.Module):
    """The model's own published eval transform.

    timm models (UNI2-h, Virchow2): built from their pretrained_cfg (resize, crop, ImageNet
    normalisation). GenBio-PathFM: the model card's transform, i.e. Resize((224, 224)),
    ToTensor and its own H&E mean/std from config.json.
    """
    if isinstance(model, PreTrainedModel):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(model.config.image_mean), std=tuple(model.config.image_std)),
        ])
    return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
