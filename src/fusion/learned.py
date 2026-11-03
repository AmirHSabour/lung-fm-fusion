"""Learned fusion (N1–N3): a small network mixes one stream per block.

Each block (each image model, and age and sex) is standardised and projected to `d`
numbers, `Linear → GELU → Dropout`; the streams are mixed by a gate, by self-attention or by
a low-rank bilinear interaction (docs/agent.md); a linear softmax layer with the
baselines' class-weighted loss gives the prediction. Training settings are fixed (AdamW,
full batch, fixed epochs); weight decay is the regularisation tuned in Phase 2. Each fit
trains `n_seeds` networks and averages their probabilities. fp32, TF32 off, deterministic.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.fusion.blocks import Blocks

KINDS = ("gated", "attention", "bilinear")


class FusionNet(nn.Module):
    def __init__(self, widths: list[int], kind: str, d: int, dropout: float, n_classes: int):
        super().__init__()
        self.kind = kind
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(p, d), nn.GELU(), nn.Dropout(dropout))
                                   for p in widths])
        B = len(widths)
        if kind == "gated":
            self.gate = nn.Linear(B * d, B)
        elif kind == "attention":
            self.stream_emb = nn.Parameter(torch.randn(B, d) * 0.02)
            self.attn = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
        elif kind == "bilinear":
            self.A = nn.Linear((B - 1) * d, d)    # joined image streams
            self.Bm = nn.Linear(d, d)             # age-and-sex stream (always the last block, Blocks)
        else:
            raise ValueError(f"kind must be one of {KINDS}")
        self.head = nn.Linear(d, n_classes)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        h = [p(x) for p, x in zip(self.proj, xs)]
        if self.kind == "gated":
            a = torch.softmax(self.gate(torch.cat(h, dim=1)), dim=1)          # (n, B)
            z = (torch.stack(h, dim=1) * a.unsqueeze(-1)).sum(dim=1)
        elif self.kind == "attention":
            tokens = torch.stack(h, dim=1) + self.stream_emb                  # (n, B, d)
            z = self.attn(tokens, tokens, tokens, need_weights=False)[0].mean(dim=1)
        else:
            z = self.A(torch.cat(h[:-1], dim=1)) * self.Bm(h[-1])
        return self.head(z)


class LearnedFusion(BaseEstimator, ClassifierMixin):
    def __init__(self, blocks: Blocks, kind: str = "gated", d: int = 32, weight_decay: float = 0.01,
                 epochs: int = 100, lr: float = 1e-3, dropout: float = 0.25, n_seeds: int = 3,
                 seed: int = 0, device: str = "cpu"):
        self.blocks = blocks
        self.kind = kind
        self.d = d
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.n_seeds = n_seeds
        self.seed = seed
        self.device = device

    def _tensors(self, X: np.ndarray) -> list[torch.Tensor]:
        return [torch.as_tensor(self.scalers_[n].transform(self.blocks.take(X, n)), dtype=torch.float32,
                                device=self.device) for n in self.blocks.names]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedFusion":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.classes_ = np.unique(y)
        self.scalers_ = {n: StandardScaler().fit(self.blocks.take(X, n)) for n in self.blocks.names}
        xs = self._tensors(X)
        y_t = torch.as_tensor(np.searchsorted(self.classes_, y), device=self.device)
        counts = np.bincount(y_t.cpu().numpy(), minlength=len(self.classes_))
        weight = torch.as_tensor(len(y) / (len(self.classes_) * counts), dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        widths = [x.shape[1] for x in xs]
        self.nets_ = []
        for s in range(self.n_seeds):
            torch.manual_seed(self.seed * 1000 + s)
            net = FusionNet(widths, self.kind, self.d, self.dropout, len(self.classes_)).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss_fn(net(xs), y_t).backward()
                opt.step()
            self.nets_.append(net.eval())
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xs = self._tensors(X)
        P = torch.stack([torch.softmax(net(xs), dim=1) for net in self.nets_]).mean(dim=0)
        return P.double().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]



# ---------------------------------------------------------------------------------------
# v02: age and sex enter as tokens from a feature tokenizer (src/fusion/metadata.py).
# The v01 classes above are unchanged.

class FusionNetTok(nn.Module):
    """Image streams plus age-and-sex tokens, mixed by a gate, attention or a bilinear term."""

    def __init__(self, widths: list[int], encoding: str | None, kind: str, d: int, dropout: float,
                 n_classes: int):
        super().__init__()
        from src.fusion.metadata import MetaTokenizer
        self.kind = kind
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(p, d), nn.GELU(), nn.Dropout(dropout))
                                   for p in widths])
        self.tok = MetaTokenizer(encoding, d, dropout) if encoding else None
        S = len(widths) + (self.tok.n_tokens if self.tok else 0)
        if kind == "gated":
            self.gate = nn.Linear(S * d, S)
        elif kind == "attention":
            self.type_emb = nn.Parameter(torch.randn(S, d) * 0.02)
            self.attn = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
        elif kind == "bilinear":
            if not widths or self.tok is None:
                raise ValueError("bilinear fusion needs image streams and age and sex")
            self.A = nn.Linear(len(widths) * d, d)
            self.Bm = nn.Linear(d, d)
        else:
            raise ValueError(f"kind must be one of {KINDS}")
        self.head = nn.Linear(d, n_classes)

    def forward(self, xs: list[torch.Tensor], meta: dict | None) -> torch.Tensor:
        h = [p(x) for p, x in zip(self.proj, xs)]
        t = self.tok(**meta) if self.tok is not None else None            # (n, T, d)
        if self.kind == "bilinear":
            z = self.A(torch.cat(h, dim=1)) * self.Bm(t.mean(dim=1))
            return self.head(z)
        streams = torch.stack(h, dim=1) if h else None
        if t is not None:
            streams = t if streams is None else torch.cat([streams, t], dim=1)    # (n, S, d)
        if self.kind == "gated":
            a = torch.softmax(self.gate(streams.flatten(1)), dim=1)
            z = (streams * a.unsqueeze(-1)).sum(dim=1)
        else:
            tokens = streams + self.type_emb
            z = self.attn(tokens, tokens, tokens, need_weights=False)[0].mean(dim=1)
        return self.head(z)


class LearnedFusionTok(BaseEstimator, ClassifierMixin):
    """v02 learned fusion: image streams + age-and-sex tokens (`encoding`)."""

    def __init__(self, blocks: Blocks, kind: str = "gated", encoding: str = "ft", d: int = 32,
                 weight_decay: float = 0.01, epochs: int = 100, lr: float = 1e-3, dropout: float = 0.25,
                 n_seeds: int = 3, seed: int = 0, device: str = "cpu"):
        self.blocks = blocks
        self.kind = kind
        self.encoding = encoding
        self.d = d
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.n_seeds = n_seeds
        self.seed = seed
        self.device = device

    def _inputs(self, X: np.ndarray) -> tuple[list[torch.Tensor], dict | None]:
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        xs = [t(self.scalers_[n].transform(self.blocks.take(X, n))) for n in self.blocks.images]
        if "M" not in self.blocks.names:
            return xs, None
        M = self.blocks.take(X, "M")
        meta = {"age_z": t((M[:, 0] - self.age_[0]) / self.age_[1]),
                "sex": t(M[:, 1].round(), torch.long),
                "sex_z": t((M[:, 1] - self.sex_[0]) / self.sex_[1]),
                "age_ple": t(self.ple_.transform(M[:, 0]))}
        return xs, meta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedFusionTok":
        from src.fusion.metadata import PiecewiseLinear
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.classes_ = np.unique(y)
        self.scalers_ = {n: StandardScaler().fit(self.blocks.take(X, n)) for n in self.blocks.images}
        if "M" in self.blocks.names:
            M = self.blocks.take(X, "M")
            self.age_ = (M[:, 0].mean(), max(M[:, 0].std(), 1e-6))
            self.sex_ = (M[:, 1].mean(), max(M[:, 1].std(), 1e-6))
            self.ple_ = PiecewiseLinear().fit(M[:, 0])
        xs, meta = self._inputs(X)
        y_t = torch.as_tensor(np.searchsorted(self.classes_, y), device=self.device)
        counts = np.bincount(y_t.cpu().numpy(), minlength=len(self.classes_))
        weight = torch.as_tensor(len(y) / (len(self.classes_) * counts), dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        widths = [x.shape[1] for x in xs]
        enc = self.encoding if meta is not None else None
        self.nets_ = []
        for s in range(self.n_seeds):
            torch.manual_seed(self.seed * 1000 + s)
            net = FusionNetTok(widths, enc, self.kind, self.d, self.dropout, len(self.classes_)).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss_fn(net(xs, meta), y_t).backward()
                opt.step()
            self.nets_.append(net.eval())
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xs, meta = self._inputs(X)
        P = torch.stack([torch.softmax(net(xs, meta), dim=1) for net in self.nets_]).mean(dim=0)
        return P.double().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
