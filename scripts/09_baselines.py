"""Stage 09: rows 1-5 of the README comparison table on the development set, per seed.

Rows: metadata only; each foundation model alone; best single model + metadata (best by
development macro AUROC). Every row uses the same classifier and tuning budget
(src/fusion/classifier.py). Scores are out-of-fold over the 8 committed folds; the test
set is never read. If a model's features are missing, its row is skipped and row 5 is
marked provisional.

Usage (from the repo root):
    uv run python -m scripts.09_baselines [--seeds 0 1]
"""
from __future__ import annotations

import os

# One BLAS thread per fit: the inner CV already runs fits in parallel, and letting each fit
# also use every core oversubscribes the CPU and slows the stage down more than 12-fold.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.cv import run_cv
from src.eval.dataset import load_dev
from src.eval.metrics import bootstrap_ci, confusion, per_class_auroc
from src.fusion.classifier import tuned_logistic
from src.utils.config import load_config

META_ROW, BEST_ROW = "metadata", "best_single_plus_metadata"


def evaluate(X: np.ndarray, dev, cfg: dict, seed: int) -> dict:
    c = cfg["classifier"]
    res = run_cv(X, dev.y, dev.folds, lambda: tuned_logistic(c["C_grid"], c["inner_folds"], seed, c["max_iter"]),
                 n_classes=len(dev.classes))
    P, pf = res["P"], res["per_fold"]
    out = dict(n_patients=len(dev.y), n_features=int(X.shape[1]),
               chosen_C=[f["best_params"]["lr__C"] for f in pf], per_fold=[
                   {k: v for k, v in f.items() if k != "best_params"} for f in pf],
               per_class_auroc=dict(zip(dev.classes, per_class_auroc(dev.y, P).round(4).tolist())),
               confusion=confusion(dev.y, P).tolist())
    for m in ("macro_auroc", "balanced_accuracy"):
        est, lo, hi = bootstrap_ci(dev.y, P, m, seed=seed)
        fold_vals = [f[m] for f in pf]
        out[m] = dict(oof=est, ci_low=lo, ci_high=hi, fold_mean=float(np.mean(fold_vals)),
                      fold_sd=float(np.std(fold_vals, ddof=1)))
    return out, P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/train.yaml"))
    ap.add_argument("--seeds", nargs="+", type=int)
    args = ap.parse_args()
    cfg, data = load_config(args.config), load_config("conf/data.yaml")
    feats = Path(cfg["features_dir"])
    models = [m for m in cfg["models"] if (feats / f"{m}.npz").is_file()]
    provisional = len(models) < len(cfg["models"])
    if provisional:
        print(f"features missing for {sorted(set(cfg['models']) - set(models))}: row 5 is provisional")

    rows = []
    for seed in args.seeds or data["split"]["seeds"]:
        dev = load_dev(seed, models, splits_dir=Path(data["splits_dir"]), cohort_csv=Path(data["cohort_csv"]),
                       features_dir=feats)
        out_dir = Path(cfg["runs_dir"]) / f"seed_{seed}" / "baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        inputs = {META_ROW: dev.meta} | {m: dev.blocks[m] for m in models}
        results = {}
        for name, X in inputs.items():
            results[name] = evaluate(X, dev, cfg, seed)
        best = max(models, key=lambda m: (results[m][0]["macro_auroc"]["oof"],
                                          results[m][0]["balanced_accuracy"]["oof"]))
        results[BEST_ROW] = evaluate(np.hstack([dev.blocks[best], dev.meta]), dev, cfg, seed)

        for name, (res, P) in results.items():
            res |= dict(seed=seed, row=name, model=best if name == BEST_ROW else name,
                        provisional=provisional and name == BEST_ROW,
                        classifier=cfg["classifier"], models_available=models)
            (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
            np.savez(out_dir / f"{name}.npz", P=P, y=dev.y, WSI_ID=dev.wsi_ids)
            rows.append(dict(seed=seed, row=name, model=res["model"], provisional=res["provisional"],
                             macro_auroc=res["macro_auroc"]["oof"], auroc_ci_low=res["macro_auroc"]["ci_low"],
                             auroc_ci_high=res["macro_auroc"]["ci_high"], auroc_fold_sd=res["macro_auroc"]["fold_sd"],
                             balanced_accuracy=res["balanced_accuracy"]["oof"],
                             bacc_ci_low=res["balanced_accuracy"]["ci_low"],
                             bacc_ci_high=res["balanced_accuracy"]["ci_high"]))
            print(f"seed {seed} {name:28s} macro AUROC {res['macro_auroc']['oof']:.3f} "
                  f"[{res['macro_auroc']['ci_low']:.3f}, {res['macro_auroc']['ci_high']:.3f}]  "
                  f"bal. acc. {res['balanced_accuracy']['oof']:.3f}  C {res['chosen_C']}", flush=True)

    summary = pd.DataFrame(rows)
    out = Path(cfg["runs_dir"]) / "baselines_dev.csv"
    summary.round(4).to_csv(out, index=False)
    agg = summary.groupby("row", sort=False)[["macro_auroc", "balanced_accuracy"]].agg(["mean", "std"]).round(3)
    print(agg.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
