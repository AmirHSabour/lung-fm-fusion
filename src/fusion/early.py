"""Early fusion (E1–E4): join all blocks after per-block preprocessing, one classifier.

The classifier is the baselines' class-weighted multinomial logistic regression; the
standardisation is done per block by `BlockTransformer`, so no second scaler is used.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.fusion.blocks import Blocks, BlockTransformer


def early_fusion(blocks: Blocks, l2: bool, pca: bool, k: int = 32, C: float = 0.01,
                 max_iter: int = 5000) -> Pipeline:
    return Pipeline([("blocks", BlockTransformer(blocks, l2=l2, pca=pca, k=k)),
                     ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter))])
