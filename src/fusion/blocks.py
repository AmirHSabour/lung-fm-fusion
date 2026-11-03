"""Input blocks for fusion: one block per foundation model plus the age-and-sex block.

All recipes take one matrix holding the blocks side by side; `Blocks` records which columns
belong to which source. `BlockTransformer` applies the early-fusion preprocessing per block
(docs/agent.md): optional row L2 and PCA on image blocks, standardisation, and block
scaling so that every block carries the same total variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

META = "M"
ID = "ID"          # v04: row of each patient in the tile cache; read only by multiple-instance recipes


@dataclass(frozen=True)
class Blocks:
    """Block names and their column ranges in the joined matrix; age and sex, if present, is last."""
    names: tuple[str, ...]
    bounds: tuple[tuple[int, int], ...]

    @classmethod
    def from_sizes(cls, sizes: dict[str, int]) -> "Blocks":
        names, bounds, start = [], [], 0
        for name, p in sizes.items():
            names.append(name)
            bounds.append((start, start + p))
            start += p
        if META in names and names[-1] != META:
            raise ValueError(f"the age-and-sex block {META!r} must be the last block")
        return cls(tuple(names), tuple(bounds))

    def __len__(self) -> int:
        return len(self.names)

    def slice(self, name: str) -> slice:
        return slice(*self.bounds[self.names.index(name)])

    def take(self, X: np.ndarray, name: str) -> np.ndarray:
        return X[:, self.slice(name)]

    @property
    def images(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if n not in (META, ID))

    @property
    def features(self) -> tuple[str, ...]:
        """Every block that is a model input (all but the cache-index block `ID`)."""
        return tuple(n for n in self.names if n != ID)

    def subset(self, names: list[str]) -> "Blocks":
        """The same layout restricted to `names` (columns renumbered); used with `columns`."""
        return Blocks.from_sizes({n: self.bounds[self.names.index(n)][1] - self.bounds[self.names.index(n)][0]
                                  for n in names})

    def columns(self, names: list[str]) -> np.ndarray:
        return np.concatenate([np.arange(*self.bounds[self.names.index(n)]) for n in names])


def l2_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class BlockTransformer(BaseEstimator, TransformerMixin):
    """Per block: [row L2] → [PCA to k] → standardise → scale by √(P / (B·p_b)).

    L2 and PCA apply to image blocks only. After scaling every block has total variance P/B
    and the whole matrix keeps total variance P, as plain standardisation would.
    """

    def __init__(self, blocks: Blocks, l2: bool = False, pca: bool = False, k: int = 32):
        self.blocks = blocks
        self.l2 = l2
        self.pca = pca
        self.k = k

    def _prepare(self, X: np.ndarray, name: str) -> np.ndarray:
        Xb = self.blocks.take(X, name)
        if name != META and self.l2:
            Xb = l2_rows(Xb)
        return Xb

    def fit(self, X: np.ndarray, y=None) -> "BlockTransformer":
        self.pca_, self.scaler_, widths = {}, {}, {}
        for name in self.blocks.features:
            Xb = self._prepare(X, name)
            if name != META and self.pca:
                if self.k >= len(X):
                    raise ValueError(f"PCA size {self.k} needs more than {len(X)} training patients")
                self.pca_[name] = PCA(n_components=self.k, svd_solver="full").fit(Xb)
                Xb = self.pca_[name].transform(Xb)
            self.scaler_[name] = StandardScaler().fit(Xb)
            widths[name] = Xb.shape[1]
        total = sum(widths.values())
        self.scale_ = {n: float(np.sqrt(total / (len(self.blocks.features) * w))) for n, w in widths.items()}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for name in self.blocks.features:
            Xb = self._prepare(X, name)
            if name in self.pca_:
                Xb = self.pca_[name].transform(Xb)
            parts.append(self.scaler_[name].transform(Xb) * self.scale_[name])
        return np.hstack(parts)
