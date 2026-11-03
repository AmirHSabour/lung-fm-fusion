"""Stage 12: final evaluation of the frozen designs on one seed (decisions/SUMMARY.md §6). READS THE TEST SET.

For every design in `conf/final.yaml` and every row (fused + age and sex; each model alone; best single
model + age and sex; age and sex alone; fused without age and sex), all with the design's own recipe:
- development: 8-fold out-of-fold predictions on the seed's 162 development patients (as in the loop);
- test: one model fitted on all 162 development patients predicts the seed's 41 test patients once.
The original rows 1-5 (stage 09: mean-pooled slide vectors + tuned logistic regression) are fitted on all
development patients and score the same test patients.

Nothing is selected on test: the designs, the primary design, the rows and the claim rule are frozen in
`conf/final.yaml`; "best single model" is chosen on development out-of-fold AUROC. Test labels are used
only to score. Outputs in `artifacts/final/seed_<s>/`: `results.json` (all scores and comparisons) and
`predictions.npz` (development out-of-fold and test probabilities of every row).

Usage (from the repo root; one seed per GPU):
    uv run --no-sync python -m scripts.12_final_eval --seed 0 --device cuda:0
    uv run --no-sync python -m scripts.12_final_eval --seed 0 --device cuda:0 --dry-run   # no test set read

`--dry-run` checks the whole path without the test set: development fold 0 stands in for the test
set and folds 1-7 are the development set; outputs go to `artifacts/final_dryrun/` and are never reported.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")            # one BLAS thread per fit (docs/methods.md D21)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS for the in-process test fits

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from src.agent.evaluate import compare, cross_validate, make_estimator
from src.agent.store import code_version
from src.data.cohort import CLASSES
from src.data.splits import load_split
from src.embeddings.tile_cache import index as cache_index
from src.eval.dataset import DevSet, load_dev, load_test
from src.eval.metrics import (balanced_accuracy, bootstrap_ci, confusion, macro_auroc, per_class_auroc,
                              stratified_resamples)
from src.fusion.blocks import ID, META, Blocks
from src.fusion.classifier import tuned_logistic
from src.utils.config import load_config
from src.utils.gpu import gpu_name

BASELINE_ROWS = ("metadata", "uni2h", "virchow2", "genbio_pathfm", "best_single_plus_metadata")   # stage 09


def check_gpu(device: str, gpu_type: str) -> str:
    """The GPU's name; a note is printed if it is not the type the reported results used."""
    return gpu_name(device, gpu_type)


def matrix(ds: DevSet, models: list[str], pos: dict[str, int]) -> np.ndarray:
    """[slide vectors of each model | tile-cache row | age, sex], the block layout of stage 10."""
    return np.hstack([ds.blocks[m] for m in models] + [np.array([[pos[w]] for w in ds.wsi_ids], float), ds.meta])


def with_id(names: list[str]) -> list[str]:
    return [n for n in names if n != META] + [ID] + ([META] if META in names else [])


def scores(y: np.ndarray, P: np.ndarray, idx: np.ndarray) -> dict:
    out = {"per_class_auroc": dict(zip(CLASSES, per_class_auroc(y, P).round(4).tolist())),
           "confusion": confusion(y, P).tolist()}
    for m in ("macro_auroc", "balanced_accuracy"):
        est, lo, hi = bootstrap_ci(y, P, m, idx=idx)
        out[m] = {"estimate": est, "ci": [lo, hi]}
    return out


def point(y: np.ndarray, P: np.ndarray) -> dict:
    return {"macro_auroc": macro_auroc(y, P), "balanced_accuracy": balanced_accuracy(y, P)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", type=Path, default=Path("conf/final.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="development fold 0 stands in for the test set")
    ap.add_argument("--out-dir", type=Path, help="output root (default: out_dir of the config); a re-run can write "
                                                 "elsewhere and be compared with the committed results")
    args = ap.parse_args()
    fcfg = load_config(args.config)
    cfg, train, data = load_config(fcfg["search_config"]), load_config("conf/train.yaml"), load_config("conf/data.yaml")
    if args.seed not in fcfg["seeds"]:
        raise SystemExit(f"seed {args.seed} is not in {fcfg['seeds']}")
    code = code_version()
    if code.endswith("-dirty"):
        raise SystemExit("commit all code and configuration first: the results name the exact code version")
    gpu = check_gpu(args.device, train["gpu_type"])
    models = fcfg["models"]
    root = args.out_dir or Path(fcfg["out_dir"] + ("_dryrun" if args.dry_run else ""))
    out = root / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg |= {"device": args.device, "expert_cache": str(out / "work" / "experts")}

    # Data: development and test patients of this seed, disjoint by slide and by patient.
    kw = dict(splits_dir=Path(data["splits_dir"]), cohort_csv=Path(data["cohort_csv"]),
              features_dir=Path(train["features_dir"]))
    if args.dry_run:                     # fold 0 stands in for the test set; the real test set is not read
        full = load_dev(args.seed, models, **kw)
        part = lambda m, folds: dataclasses.replace(
            full, wsi_ids=full.wsi_ids[m], blocks={k: v[m] for k, v in full.blocks.items()}, meta=full.meta[m],
            y=full.y[m], folds=folds)
        held = full.folds == 0
        dev, test = part(~held, full.folds[~held] - 1), part(held, np.full(held.sum(), -1))
    else:
        dev, test = load_dev(args.seed, models, **kw), load_test(args.seed, models, **kw)
    patient = load_split(Path(data["splits_dir"]), args.seed).set_index("WSI_ID")["SampleNumber"]
    if set(dev.wsi_ids) & set(test.wsi_ids) or set(patient[dev.wsi_ids]) & set(patient[test.wsi_ids]):
        raise SystemExit("development and test patients overlap")
    if set(np.unique(dev.y)) != set(range(len(CLASSES))) or set(np.unique(test.y)) != set(range(len(CLASSES))):
        raise SystemExit("every class must be present in development and test")
    pos = cache_index(cfg["tile_cache"])
    Xd, Xt = matrix(dev, models, pos), matrix(test, models, pos)
    blocks = Blocks.from_sizes({**{m: dev.blocks[m].shape[1] for m in models}, ID: 1, META: dev.meta.shape[1]})
    yd, yt = dev.y, test.y
    idx_d = stratified_resamples(yd, fcfg["n_resamples"], args.seed)
    idx_t = stratified_resamples(yt, fcfg["n_resamples"], args.seed)
    keep = ~np.isin(test.wsi_ids, fcfg["conflicting_label_slides"])          # D11 sensitivity analysis
    cap_s = cfg["search"]["cap_minutes"] * 60

    results = {"seed": args.seed, "code": code, "device": gpu, "config": str(args.config),
               "n_dev": int(len(yd)), "n_test": int(len(yt)), "test_wsi": test.wsi_ids.tolist(),
               "conflicting_in_test": test.wsi_ids[~keep].tolist(), "designs": {}, "baselines": {}}
    preds: dict[str, np.ndarray] = {"y_dev": yd, "y_test": yt, "wsi_dev": dev.wsi_ids, "wsi_test": test.wsi_ids}

    def run_row(design: str, spec: dict, label: str, names: list[str]) -> dict:
        names = with_id(names)
        cols, sub = blocks.columns(names), blocks.subset(names)
        t0 = time.time()
        status, Pd, _, _ = cross_validate(spec, Xd[:, cols], yd, dev.folds, sub, cfg, args.seed, args.device,
                                          out / "work", cap_s)
        if status != "done":
            raise SystemExit(f"{design}/{label}: development CV {status}")
        est = make_estimator(spec, sub, cfg, args.seed, args.device)
        Pt = est.fit(Xd[:, cols], yd).predict_proba(Xt[:, cols])        # all development patients -> test, once
        if Pt.shape != (len(yt), len(CLASSES)) or not np.allclose(Pt.sum(axis=1), 1, atol=1e-6):
            raise SystemExit(f"{design}/{label}: invalid test probabilities")
        preds[f"{design}/{label}/dev"], preds[f"{design}/{label}/test"] = Pd, Pt
        r = {"blocks": names, "dev": scores(yd, Pd, idx_d), "test": scores(yt, Pt, idx_t),
             "test_without_conflicting": point(yt[keep], Pt[keep]), "seconds": round(time.time() - t0, 1)}
        print(f"[seed {args.seed}] {design}/{label}: dev {r['dev']['macro_auroc']['estimate']:.4f}  "
              f"test {r['test']['macro_auroc']['estimate']:.4f}  ({r['seconds']:.0f} s)", flush=True)
        return r

    # The frozen designs, each with every comparison row under its own recipe.
    for design, d in fcfg["designs"].items():
        spec = {"candidate": d["candidate"], "mode": "screen"}
        rows = {"fused": run_row(design, spec, "fused", models + [META])}
        if args.seed == 0 and not args.dry_run and abs(rows["fused"]["dev"]["macro_auroc"]["estimate"] - d["seed0_dev_auroc"]) > 5e-4:
            raise SystemExit(f"{design}: seed 0 development AUROC "
                             f"{rows['fused']['dev']['macro_auroc']['estimate']:.4f} does not reproduce the loop's "
                             f"{d['seed0_dev_auroc']}")
        rows["no_meta"] = run_row(design, spec, "no_meta", models)
        for m in models:
            rows[m] = run_row(design, spec, m, [m])
        best = max(models, key=lambda m: rows[m]["dev"]["macro_auroc"]["estimate"])    # chosen on development
        rows["best_plus_meta"] = run_row(design, spec, "best_plus_meta", [best, META]) | {"model": best}
        rows["meta"] = run_row(design, spec, "meta", [META])
        Pf = preds[f"{design}/fused/test"]
        comparisons = {label: compare(yt, Pf, preds[f"{design}/{label}/test"], idx_t)
                       for label in rows if label != "fused"}
        results["designs"][design] = {"candidate": d["candidate"], "role": d["role"], "rows": rows,
                                      "fused_vs_row_on_test": comparisons}

    # The original rows 1-5 (stage 09), refitted on all development patients and scored on the same test set.
    c = train["classifier"]
    base_dir = Path(train["runs_dir"]) / f"seed_{args.seed}" / "baselines"
    best_model = json.loads((base_dir / "best_single_plus_metadata.json").read_text())["model"]
    inputs = {"metadata": (dev.meta, test.meta)} | {m: (dev.blocks[m], test.blocks[m]) for m in models}
    inputs["best_single_plus_metadata"] = (np.hstack([dev.blocks[best_model], dev.meta]),
                                           np.hstack([test.blocks[best_model], test.meta]))
    for row in BASELINE_ROWS:
        f = np.load(base_dir / f"{row}.npz", allow_pickle=True)
        at = {w: i for i, w in enumerate(f["WSI_ID"])}
        Pd = f["P"][[at[w] for w in dev.wsi_ids]]                               # stage 09 out-of-fold predictions
        Ad, At = inputs[row]
        Pt = tuned_logistic(c["C_grid"], c["inner_folds"], args.seed, c["max_iter"]).fit(Ad, yd).predict_proba(At)
        preds[f"baseline/{row}/dev"], preds[f"baseline/{row}/test"] = Pd, Pt
        results["baselines"][row] = {"model": best_model if row == "best_single_plus_metadata" else row,
                                     "dev": scores(yd, Pd, idx_d), "test": scores(yt, Pt, idx_t),
                                     "test_without_conflicting": point(yt[keep], Pt[keep])}
        print(f"[seed {args.seed}] baseline/{row}: dev {results['baselines'][row]['dev']['macro_auroc']['estimate']:.4f}"
              f"  test {results['baselines'][row]['test']['macro_auroc']['estimate']:.4f}", flush=True)
    for design in fcfg["designs"]:
        Pf = preds[f"{design}/fused/test"]
        results["designs"][design]["fused_vs_baseline_on_test"] = {
            row: compare(yt, Pf, preds[f"baseline/{row}/test"], idx_t) for row in BASELINE_ROWS}

    np.savez(out / "predictions.npz", **preds)
    np.save(out / "test_bootstrap_idx.npy", idx_t)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
