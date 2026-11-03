"""Tile embedding with a frozen encoder.

Tiles are read from the resampled 0.5 microns-per-pixel slide at the positions chosen by
stage 06, passed through the model's own published transform, and encoded in inference
mode. Nothing is trained here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvips
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TileDataset(Dataset):
    """Tiles of one slide; each DataLoader worker opens its own handle to the slide."""

    def __init__(self, slide: Path, xy: np.ndarray, tile_px: int, transform):
        self.slide, self.xy, self.tile_px, self.transform = str(slide), xy, tile_px, transform
        self._im = None

    def __len__(self) -> int:
        return len(self.xy)

    def __getitem__(self, i: int) -> torch.Tensor:
        if self._im is None:
            self._im = pyvips.Image.new_from_file(self.slide, access="random")
        x, y = (int(v) for v in self.xy[i])
        tile = self._im.crop(x, y, self.tile_px, self.tile_px).numpy()
        return self.transform(Image.fromarray(tile))


def encoder_outputs(key: str, out: torch.Tensor) -> dict[str, torch.Tensor]:
    """Model-specific tile embeddings, as recommended by each model card.

    uni2h:    the pooled class token (1536).
    genbio_pathfm: the class tokens of the R, G and B channels, concatenated (3 x 1536 = 4608).
    virchow2: class token (1280) and mean of the 256 patch tokens (1280), kept separately;
              their concatenation (2560) is the recommended tile embedding, and the class
              token alone is Prism2's input.
    """
    if key == "uni2h":
        return {"cls": out}
    if key == "virchow2":
        return {"cls": out[:, 0], "patch_mean": out[:, 5:].mean(dim=1)}
    if key == "genbio_pathfm":
        return {"cls": out}
    raise KeyError(key)


@torch.inference_mode()
def embed_slide(model: torch.nn.Module, key: str, transform, slide: Path, xy: np.ndarray, *,
                tile_px: int = 224, batch_size: int = 256, workers: int = 16,
                device: str = "cuda", precision: str = "fp16") -> dict[str, np.ndarray]:
    """Embeddings of every tile at xy, as float32 arrays of shape (n_tiles, dim).

    precision "fp32" runs the model in true full precision (TF32 tensor-core shortcuts are
    switched off); "fp16" runs it under fp16 autocast.
    """
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # "spawn", not the Linux default "fork": libvips keeps a thread pool, and forking a
    # process that has used it can copy a held lock into the workers, which then deadlock.
    loader = DataLoader(TileDataset(slide, xy, tile_px, transform), batch_size=batch_size,
                        num_workers=workers, pin_memory=True, persistent_workers=False,
                        multiprocessing_context="spawn" if workers > 0 else None)
    chunks: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=precision == "fp16"):
            out = model(batch.to(device, non_blocking=True))
        for name, t in encoder_outputs(key, out).items():
            chunks.setdefault(name, []).append(t.float().cpu().numpy())
    return {name: np.concatenate(c) for name, c in chunks.items()}
