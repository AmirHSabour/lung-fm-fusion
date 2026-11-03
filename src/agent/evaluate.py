"""One experiment: a recipe with one setting, scored by the 8-fold development CV.

The cross-validation runs in a separate process so the wall-clock cap can stop it
(docs/agent.md). Scores and paired comparisons use the seed's stored bootstrap
resamples, so every comparison of a seed is made on the same resamples.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from src.eval.cv import run_cv
from src.eval.metrics import bootstrap_ci, confusion, paired_bootstrap, per_class_auroc
from src.fusion.blocks import Blocks
from src.fusion.registry import build


def make_estimator(spec: dict, blocks: Blocks, cfg: dict, seed: int, device: str):
    """v01 specs: {recipe, phase, size}; v02+ specs: {candidate, mode}."""
    if "candidate" in spec:
        from src.fusion.candidates import build_candidate
        return build_candidate(spec["candidate"], spec["mode"], blocks, cfg, seed, device)
    return build(spec["recipe"], blocks, spec["phase"], spec["size"], cfg, seed, device)


def _child(spec: dict, X: np.ndarray, y: np.ndarray, folds: np.ndarray,
           blocks: Blocks, cfg: dict, seed: int, device: str, out: str) -> None:
    res = run_cv(X, y, folds, lambda: make_estimator(spec, blocks, cfg, seed, device), n_classes=len(np.unique(y)))
    np.save(out + ".P.npy", res["P"])
    Path(out + ".folds.json").write_text(json.dumps(res["per_fold"], default=float))


def cross_validate(spec: dict, X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                   blocks: Blocks, cfg: dict, seed: int, device: str, work: Path,
                   cap_s: float) -> tuple[str, np.ndarray | None, list | None, float]:
    """(status, out-of-fold probabilities, per-fold scores, seconds); status is done, timed_out or failed."""
    work.mkdir(parents=True, exist_ok=True)
    out = str(work / "running")
    for suffix in (".P.npy", ".folds.json"):
        Path(out + suffix).unlink(missing_ok=True)
    t0 = time.time()
    proc = mp.get_context("spawn").Process(target=_child, args=(spec, X, y, folds, blocks, cfg, seed, device, out))
    proc.start()
    proc.join(cap_s)
    seconds = time.time() - t0
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "timed_out", None, None, seconds
    if proc.exitcode != 0 or not Path(out + ".P.npy").exists():
        return "failed", None, None, seconds
    P = np.load(out + ".P.npy")
    per_fold = json.loads(Path(out + ".folds.json").read_text())
    return "done", P, per_fold, seconds


def compare(y: np.ndarray, P_a: np.ndarray, P_b: np.ndarray, idx: np.ndarray) -> dict:
    """A versus B on the same patients and resamples: AUROC gain, its 95% CI, share of resamples A wins."""
    pb = paired_bootstrap(y, P_a, P_b, "macro_auroc", idx=idx)
    return {"gain": pb["gain"], "ci": [pb["ci_low"], pb["ci_high"]], "share_won": pb["p_better"]}


def scores(y: np.ndarray, P: np.ndarray, per_fold: list, idx: np.ndarray, classes: tuple[str, ...]) -> dict:
    out = {"per_class_auroc": dict(zip(classes, per_class_auroc(y, P).round(4).tolist())),
           "confusion": confusion(y, P).tolist(),
           "chosen_per_fold": [f["best_params"] for f in per_fold]}
    for m in ("macro_auroc", "balanced_accuracy"):
        est, lo, hi = bootstrap_ci(y, P, m, idx=idx)
        vals = [f[m] for f in per_fold]
        out[m] = {"oof": est, "ci": [lo, hi], "fold_mean": float(np.mean(vals)), "fold_sd": float(np.std(vals, ddof=1))}
    return out

