"""GPU identification. Our results were produced on fixed GPU types (docs/methods.md D18):
embeddings on an NVIDIA RTX PRO 6000 Blackwell, all training on an NVIDIA GeForce RTX 4090,
in fp32 with TF32 off. Other GPUs run the same code; their numbers may differ in the last digits."""
from __future__ import annotations


def gpu_name(device: str, expected: str | None = None) -> str:
    """Name of the GPU at `device`; prints a short note if it is not the `expected` type."""
    import torch
    name = torch.cuda.get_device_name(device)
    if expected and name != expected:
        print(f"Note: {device} is {name}; the reported results used {expected}. "
              "Numbers may differ slightly (docs/methods.md D18).", flush=True)
    return name
