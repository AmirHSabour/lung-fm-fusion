# Foundation model weights

Populated by `uv run python -m scripts.00_download_models`, at the Hub revisions pinned
in `conf/models.yaml`. This directory is git-ignored: the weights are large and their
licences forbid redistribution (UNI2-h, Virchow2, Prism2: CC-BY-NC-ND-4.0; GenBio-PathFM:
GenBio AI Community License, non-commercial). UNI2-h, Virchow2 and Prism2 are gated: each
person must request access on Hugging Face with their own account. Models used: UNI2-h,
Virchow2 and GenBio-PathFM; Prism2 is not used (docs/methods.md D19).

| Folder      | Hub repo              | Type                         | Embedding        | Input                         | Normalization        | Size   |
| ----------- | --------------------- | ---------------------------- | ---------------- | ----------------------------- | -------------------- | ------ |
| `UNI2-h/`   | `MahmoodLab/UNI2-h`   | Tile encoder (timm ViT-H/14) | 1536 (CLS)       | 224x224, bilinear resize      | ImageNet mean / std  | 2.7 GB |
| `Virchow2/` | `paige-ai/Virchow2`   | Tile encoder (timm ViT-H/14) | 1280 CLS (+1280 mean patch = 2560) | 224x224, bicubic resize | ImageNet mean / std | 2.5 GB |
| `GenBio-PathFM/` | `genbio-ai/genbio-pathfm` | Tile encoder (transformers, custom code; 1.1B ViT) | 4608 (CLS of each RGB channel, 3 x 1536) | 224x224 resize | own H&E mean (0.697, 0.575, 0.728) / std (0.188, 0.240, 0.187) | 4.3 GB |
| `Prism2/`   | `paige-ai/Prism2`     | Slide encoder (transformers, custom code) | 2560 base / 3072 diagnostic | Virchow2 **CLS-only** tile embeddings `(N, 1280)` at 20x | n/a | 17.9 GB |

Prism2 depends on Virchow2 tile embeddings (class token only, 20x / 0.5 mpp,
224 px tiles, background removed) and requires `flash-attn`, which is pinned in
`pyproject.toml`.

## Loading

Use `src/embeddings/models.py`, which loads each encoder from these local files **with
its own published preprocessing**. Loading timm weights with only
`pretrained_cfg_overlay=dict(file=...)` silently falls back to timm's generic defaults
(mean/std 0.5, crop_pct 0.9); the helper merges each repo's `config.json` preprocessing
back in (docs/methods.md D6). `tests/test_models.py` checks this.

```python
from pathlib import Path
from src.embeddings.models import load_uni2h, load_virchow2, model_transform

uni2 = load_uni2h(Path("models/UNI2-h"))          # (B, 3, 224, 224) -> (B, 1536)
virchow2 = load_virchow2(Path("models/Virchow2")) # (B, 3, 224, 224) -> (B, 261, 1280)
# Virchow2 tokens: [0] = class token, [1:5] = registers, [5:] = 256 patch tokens
uni2_tf, virchow2_tf = model_transform(uni2), model_transform(virchow2)

# GenBio-PathFM (runs its own model code; reviewed, docs/methods.md D19)
from src.embeddings.models import load_genbio_pathfm
genbio = load_genbio_pathfm(Path("models/GenBio-PathFM"))  # (B, 3, 224, 224) -> (B, 4608)
genbio_tf = model_transform(genbio)

# Prism2 (not used; only if access is granted later)
from transformers import AutoModel, AutoProcessor
prism2 = AutoModel.from_pretrained("models/Prism2", trust_remote_code=True, torch_dtype="auto").cuda().eval()
processor = AutoProcessor.from_pretrained("models/Prism2", trust_remote_code=True)
```
