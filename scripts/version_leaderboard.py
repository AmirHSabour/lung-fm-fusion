"""Compare the winners of all versions and estimate the global winner's curse (docs/loop.md).

For every `artifacts/versions/vNN/seed_<s>/agent/` with a `selected.yaml`: the winner's macro
AUROC with CI, balanced accuracy, gain over row 5 (paired bootstrap), and whether it becomes
the "current best" (it must clearly beat the current best: wins in >= 90% of the paired
resamples and balanced accuracy at most 0.02 lower). The global winner's curse is computed
over every finished experiment of every version on the same model set: for each resample,
the experiment that scores best on that resample, minus its score on all patients.

Usage: uv run --no-sync python -m scripts.version_leaderboard [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import balanced_accuracy, bootstrap_ci, macro_auroc, paired_bootstrap, stratified_resamples

ROOT = Path("artifacts/versions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win-share", type=float, default=0.9)
    ap.add_argument("--bacc-margin", type=float, default=0.02)
    args = ap.parse_args()
    runs = []
    for d in sorted(ROOT.glob(f"v*/seed_{args.seed}/agent")):
        sel = d / "selected.yaml"
        if not sel.exists():
            continue
        recs = [json.loads(line) for line in (d / "trials.jsonl").read_text().splitlines() if line.strip()]
        winner = yaml.safe_load(sel.read_text())["winner"]
        npz = np.load(d / "trials" / f"{winner}.npz", allow_pickle=True)
        exps = [r for r in recs if r["type"] == "experiment" and r["status"] == "done"]
        Ps = {r["id"]: np.load(d / "trials" / f"{r['id']}.npz")["P"] for r in exps
              if (d / "trials" / f"{r['id']}.npz").exists()}
        runs.append({"version": d.parts[-3], "models": tuple(recs[0]["models"]), "winner": winner,
                     "candidate": next(r for r in recs if r.get("id") == winner)["recipe"], "P": npz["P"],
                     "y": npz["y"], "wsi": npz["WSI_ID"], "Ps": Ps, "n_exp": len([r for r in recs if r["type"] == "experiment"]),
                     "n_dec": len([r for r in recs if r["type"] == "decision"])})
    if not runs:
        raise SystemExit("no finished version found")
    y, wsi = runs[0]["y"], runs[0]["wsi"]
    for r in runs:
        if not (np.array_equal(r["y"], y) and np.array_equal(r["wsi"], wsi)):
            raise SystemExit(f"{r['version']} was run on different patients")
    idx = stratified_resamples(y, 2000, args.seed)
    b = np.load(Path("artifacts/runs") / f"seed_{args.seed}" / "baselines" / "best_single_plus_metadata.npz",
                allow_pickle=True)
    pos = {w: i for i, w in enumerate(b["WSI_ID"])}
    P5 = b["P"][[pos[w] for w in wsi]]

    rows, best_by_models, pool_by_models = [], {}, {}
    for r in runs:
        P = r["P"]
        a, lo, hi = bootstrap_ci(y, P, "macro_auroc", idx=idx)
        v5 = paired_bootstrap(y, P, P5, "macro_auroc", idx=idx)
        cur = best_by_models.get(r["models"])
        becomes = cur is None
        if cur is not None:
            c = paired_bootstrap(y, P, cur["P"], "macro_auroc", idx=idx)
            becomes = c["p_better"] >= args.win_share and \
                balanced_accuracy(y, P) >= balanced_accuracy(y, cur["P"]) - args.bacc_margin
        if becomes:
            best_by_models[r["models"]] = r
        pool = pool_by_models.setdefault(r["models"], [])
        pool += list(r["Ps"].values())
        full = np.array([macro_auroc(y, p) for p in pool])
        infl = []
        for i in idx:
            on_b = np.array([macro_auroc(y[i], p[i]) for p in pool])
            k = int(on_b.argmax())
            infl.append(on_b[k] - full[k])
        rows.append({"version": r["version"], "models": "+".join(r["models"]), "winner": r["winner"],
                     "candidate": r["candidate"], "experiments": r["n_exp"], "decisions": r["n_dec"],
                     "auroc": round(a, 4), "auroc_ci": f"[{lo:.3f}, {hi:.3f}]",
                     "bacc": round(balanced_accuracy(y, P), 4), "gain_vs_row5": round(v5["gain"], 4),
                     "gain_ci": f"[{v5['ci_low']:+.3f}, {v5['ci_high']:+.3f}]", "share_won_vs_row5": v5["p_better"],
                     "current_best": best_by_models[r["models"]]["version"],
                     "global_experiments": len(pool), "global_winners_curse": round(float(np.mean(infl)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"leaderboard_seed{args.seed}.csv", index=False)
    print(f"row 5: AUROC {macro_auroc(y, P5):.3f}, balanced accuracy {balanced_accuracy(y, P5):.3f}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
