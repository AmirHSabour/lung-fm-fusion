"""Post-hoc diagnostics of a finished version run (decisions/vNN.md §5–6).

Applies one recorded experiment's candidate to subsets of the sources (matched rows) with the same
folds and resamples, and appends the results to `<run_dir>/posthoc.jsonl`. The agent's record
(`trials.jsonl`) is never touched: these rows are analysis, not search, and never enter selection.
Development patients only.

Usage (from the repo root):
    uv run --no-sync python -m scripts.11_post_hoc --seed 0 --config conf/versions/v05.yaml --of s0-T006
    uv run --no-sync python -m scripts.11_post_hoc --seed 0 --config conf/versions/v06.yaml --of s0-T006 --candidate A2:raw

With `--candidate`, a candidate the agent never ran is evaluated on all sources ("full") and on
the subsets, and compared with `--of`. Such rows are diagnostics of a pre-registered prediction,
labelled with the candidate, and never enter selection.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import json
from pathlib import Path

import numpy as np

from src.agent.evaluate import compare, cross_validate, scores
from src.agent.store import TrialStore, code_version
from src.data.cohort import CLASSES
from src.eval.dataset import load_dev
from src.fusion.blocks import ID, META, Blocks
from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--of", required=True, help="experiment ID whose candidate is applied to source subsets")
    ap.add_argument("--candidate", help="evaluate this candidate (screen) instead, compared with --of")
    args = ap.parse_args()
    cfg, train, data = load_config(args.config), load_config("conf/train.yaml"), load_config("conf/data.yaml")
    models = cfg["models"]
    run_dir = Path(cfg["out_dir"]) / f"seed_{args.seed}" / "agent"
    cfg["expert_cache"] = str(run_dir / "work" / "experts")
    rec = next(r for r in TrialStore(run_dir / "trials.jsonl", args.seed).records() if r.get("id") == args.of)
    spec = {"candidate": args.candidate or rec["candidate"], "mode": "screen" if args.candidate else rec["mode"]}
    from src.fusion.candidates import family
    fam = family(spec["candidate"])

    dev = load_dev(args.seed, models, splits_dir=Path(data["splits_dir"]), cohort_csv=Path(data["cohort_csv"]),
                   features_dir=Path(train["features_dir"]))
    sizes = {m: dev.blocks[m].shape[1] for m in models}
    parts = [dev.blocks[m] for m in models]
    if cfg.get("tile_cache"):
        from src.embeddings.tile_cache import index as cache_index
        pos_c = cache_index(cfg["tile_cache"])
        parts.append(np.array([[pos_c[w]] for w in dev.wsi_ids], dtype=float))
        sizes[ID] = 1
    X = np.hstack(parts + [dev.meta])
    blocks = Blocks.from_sizes({**sizes, META: dev.meta.shape[1]})
    y, idx = dev.y, np.load(run_dir / "bootstrap_idx.npy")
    P_ref = np.load(run_dir / "trials" / f"{args.of}.npz")["P"]

    extra = [ID] if ID in blocks.names else []
    rows = ([("full", list(models) + [META])] if args.candidate else []) + [(f"{m}_m", [m]) for m in models] + [("diag_no_meta", list(models))] + \
           [(f"{m}+meta", [m, META]) for m in models]
    out = run_dir / "posthoc.jsonl"
    tag = f"{args.of}/{spec['candidate']}" if args.candidate else args.of
    done = {(r["of"], r["label"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    for label, names in rows:
        if (tag, label) in done:
            continue
        names = [n for n in names if n != META] + extra + ([META] if META in names else [])
        status, P, per_fold, sec = cross_validate(spec, X[:, blocks.columns(names)], y, dev.folds,
                                                  blocks.subset(names), cfg, args.seed, cfg["device"],
                                                  run_dir / "work", cfg["search"]["cap_minutes"] * 60)
        r = {"of": tag, "candidate": spec["candidate"], "mode": spec["mode"], "family": fam, "label": label,
             "blocks": names, "status": status, "seconds": sec, "code": code_version()}
        if status == "done":
            r["scores"] = scores(y, P, per_fold, idx, CLASSES)
            r["vs_of"] = compare(y, P_ref, P, idx)      # share of resamples in which `of` beats this row
            np.savez(run_dir / "trials" / f"posthoc_{tag.replace('/', '_').replace(':', '-')}_{label}.npz", P=P, y=y)
            print(f"{label}: AUROC {r['scores']['macro_auroc']['oof']:.4f}, balanced accuracy "
                  f"{r['scores']['balanced_accuracy']['oof']:.3f}; {args.of} wins "
                  f"{r['vs_of']['share_won']:.3f}", flush=True)
            if label == "full":
                print("    per class " + ", ".join(f"{k} {v:.3f}" for k, v in r["scores"]["per_class_auroc"].items()), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(r, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
