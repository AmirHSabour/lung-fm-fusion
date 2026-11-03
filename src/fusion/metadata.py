"""Encodings of age and sex (decisions/v02.md §2.1).

Inputs are the two raw age-and-sex columns of the joined matrix: age in years and sex
(male = 1). Every encoding is fitted on training patients only.

Linear recipes (early, late) use non-learned encodings, because a learned embedding followed
by a linear model gives the same predictions as its input:
- `raw`: [age, sex] (v01);
- `ple`: piecewise-linear encoding of age (Gorishniy et al., 2022) plus sex.

Networks (learned recipes) use token modules that map age and sex to d-dimensional tokens:
- `joint`: one token from [age, sex] (v01);
- `ft`: FT-Transformer feature tokenizer (Gorishniy et al., 2021), one token per variable;
- `plr`: periodic embedding of age (Gorishniy et al., 2022) and a sex embedding;
- `ple`: piecewise-linear encoding of age and a sex embedding.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from torch import nn

from src.fusion.blocks import META, Blocks

LINEAR_ENCODINGS = ("raw", "ple")
TOKEN_ENCODINGS = ("joint", "ft", "plr", "ple")
PLE_BINS = 8
PLR_FREQUENCIES = 16


class PiecewiseLinear(BaseEstimator, TransformerMixin):
    """Age → T components in [0, 1]; bin edges at the training quantiles of age."""

    def __init__(self, bins: int = PLE_BINS):
        self.bins = bins

    def fit(self, a: np.ndarray, y=None) -> "PiecewiseLinear":
        edges = np.quantile(np.asarray(a, float).ravel(), np.linspace(0, 1, self.bins + 1))
        for i in range(1, len(edges)):            # strictly increasing even with tied ages
            edges[i] = max(edges[i], edges[i - 1] + 1e-6)
        self.edges_ = edges
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, float).reshape(-1, 1)
        lo, hi = self.edges_[:-1], self.edges_[1:]
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def encoded_width(encoding: str) -> int:
    return {"raw": 2, "ple": PLE_BINS + 1}[encoding]


class MetaColumns(BaseEstimator, TransformerMixin):
    """[age, sex] → the columns of a linear encoding (`raw` or `ple`)."""

    def __init__(self, encoding: str = "raw"):
        self.encoding = encoding

    def fit(self, M: np.ndarray, y=None) -> "MetaColumns":
        if self.encoding not in LINEAR_ENCODINGS:
            raise ValueError(f"linear encodings are {LINEAR_ENCODINGS}")
        if self.encoding == "ple":
            self.ple_ = PiecewiseLinear().fit(M[:, 0])
        return self

    def transform(self, M: np.ndarray) -> np.ndarray:
        if self.encoding == "raw":
            return M
        return np.hstack([self.ple_.transform(M[:, 0]), M[:, 1:2]])


def encoded_blocks(blocks: Blocks, encoding: str) -> Blocks:
    """Block layout after the age-and-sex columns are replaced by their encoding."""
    sizes = {n: b - a for n, (a, b) in zip(blocks.names, blocks.bounds)}
    if META in sizes:
        sizes[META] = encoded_width(encoding)
    return Blocks.from_sizes(sizes)


class MetaEncoded(BaseEstimator, ClassifierMixin):
    """Replace the age-and-sex columns by a linear encoding, then fit `inner` (built on the
    encoded layout). The encoding is fitted on the training patients of each fit."""

    def __init__(self, inner: BaseEstimator, blocks: Blocks, encoding: str = "raw"):
        self.inner = inner
        self.blocks = blocks
        self.encoding = encoding

    def _encode(self, X: np.ndarray) -> np.ndarray:
        if META not in self.blocks.names:
            return X
        sl = self.blocks.slice(META)
        return np.hstack([X[:, :sl.start], self.enc_.transform(X[:, sl]), X[:, sl.stop:]])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaEncoded":
        if META in self.blocks.names:
            self.enc_ = MetaColumns(self.encoding).fit(self.blocks.take(X, META))
        self.inner_ = clone(self.inner).fit(self._encode(X), y)
        self.classes_ = self.inner_.classes_
        self.best_params_ = getattr(self.inner_, "best_params_", None)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.inner_.predict_proba(self._encode(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class MetaTokenizer(nn.Module):
    """Age and sex → d-dimensional tokens, shape (n, n_tokens, d).

    Forward inputs: `age_z` (standardised age, (n,)), `sex` (0/1 integers, (n,)), `sex_z`
    (standardised sex, used by `joint` as in v01), `age_ple` (piecewise-linear encoding of
    age, (n, T); used by `ple` only).
    """

    def __init__(self, encoding: str, d: int, dropout: float):
        super().__init__()
        if encoding not in TOKEN_ENCODINGS:
            raise ValueError(f"token encodings are {TOKEN_ENCODINGS}")
        self.encoding = encoding
        if encoding == "joint":
            self.joint = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Dropout(dropout))
            return
        self.sex = nn.Embedding(2, d)
        self.sex_bias = nn.Parameter(torch.zeros(d))
        if encoding == "ft":
            self.age_w = nn.Parameter(torch.randn(d) / d ** 0.5)
            self.age_b = nn.Parameter(torch.zeros(d))
        elif encoding == "plr":
            self.freq = nn.Parameter(torch.randn(PLR_FREQUENCIES))
            self.age_lin = nn.Linear(2 * PLR_FREQUENCIES, d)
        else:
            self.age_lin = nn.Linear(PLE_BINS, d)

    @property
    def n_tokens(self) -> int:
        return 1 if self.encoding == "joint" else 2

    def forward(self, age_z: torch.Tensor, sex: torch.Tensor, sex_z: torch.Tensor,
                age_ple: torch.Tensor) -> torch.Tensor:
        if self.encoding == "joint":
            return self.joint(torch.stack([age_z, sex_z], dim=1)).unsqueeze(1)
        if self.encoding == "ft":
            age = age_z[:, None] * self.age_w + self.age_b
        elif self.encoding == "plr":
            z = 2 * torch.pi * age_z[:, None] * self.freq
            age = torch.relu(self.age_lin(torch.cat([torch.sin(z), torch.cos(z)], dim=1)))
        else:
            age = torch.relu(self.age_lin(age_ple))
        sex_tok = self.sex(sex.long()) + self.sex_bias
        return torch.stack([age, sex_tok], dim=1)
