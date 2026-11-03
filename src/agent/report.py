"""Human-readable outputs generated from `trials.jsonl` (docs/agent.md).

Nothing here is written by hand: `summary.csv`, `decision_log.md`, `progress.png` and
`selected.yaml` are rebuilt from the record, so they always agree with it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.eval.metrics import macro_auroc  # noqa: E402

# Reference categorical palette, slots 1-2 (dataviz skill), light surface; marker shape is
# the second channel, so phase is never told by colour alone.
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
PHASE_STYLE = {1: ("#2a78d6", "o", "Fixed values (Phase 1 / screen)"), 2: ("#eb6834", "s", "Tuned by inner CV (Phase 2)")}


def summary(records: list[dict]) -> pd.DataFrame:
    rows = []
    for e in (r for r in records if r["type"] in ("experiment", "ablation", "matched")):
        sc = e.get("scores") or {}
        a, b = sc.get("macro_auroc", {}), sc.get("balanced_accuracy", {})
        rows.append({"id": e["id"], "type": e["type"], "label": e.get("label"), "phase": e.get("phase"),
                     "mode": e.get("mode"), "recipe": e["recipe"],
                     "size": e.get("size"), "status": e["status"], "macro_auroc": a.get("oof"),
                     "auroc_ci_low": (a.get("ci") or [None])[0], "auroc_ci_high": (a.get("ci") or [None, None])[1],
                     "balanced_accuracy": b.get("oof"),
                     "gain_vs_row5": (e.get("vs_row5") or {}).get("gain"),
                     "share_won_vs_row5": (e.get("vs_row5") or {}).get("share_won"),
                     "seconds": round(e["seconds"], 1)})
    return pd.DataFrame(rows)


def decision_log(records: list[dict], row5: dict) -> str:
    header = records[0]
    lines = [f"# Decision log, seed {header['seed']} ({header['mode'].replace('_', ' ')})", "",
             f"Models: {', '.join(header['models'])} + age and sex. Code {header['code']}.", "",
             f"Reference, row 5 (best single model + age and sex, plain concatenation): "
             f"macro AUROC {row5['auroc']:.3f}, balanced accuracy {row5['bacc']:.3f}.", ""]
    for r in records[1:]:
        if r["type"] == "decision":
            lines += [f"## {r['id']} Decision: {r['rule']}", "", r["reason"], "",
                      f"Based on: {', '.join(r['based_on'])}", ""]
            fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
            for k, v in r["statistics"].items():
                lines.append(f"- {k}: " + (", ".join(f"{a} {fmt(b)}" for a, b in v.items())
                                           if isinstance(v, dict) else fmt(v)))
            lines += ["", f"Action: `{json.dumps(r['action'])}`", ""]
            continue
        kind = {"ablation": "Ablation (row 5b)", "matched": f"Matched baseline {r.get('label')}"}.get(r["type"], "Experiment")
        title = f"## {r['id']} {kind}: {r['recipe']}"
        title += (f", {r['mode']}" if r.get("mode") else f", phase {r['phase']}") + (f", size {r['size']}" if r.get("size") else "")
        lines += [title, "", f"Why: {r['reason']}" + (f" (based on {', '.join(r['based_on'])})" if r["based_on"] else ""), ""]
        if r["status"] != "done":
            lines += [f"Status: **{r['status']}** after {r['seconds']:.0f} s.", ""]
            continue
        a, b = r["scores"]["macro_auroc"], r["scores"]["balanced_accuracy"]
        lines.append(f"Macro AUROC {a['oof']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}], folds {a['fold_mean']:.3f} ± "
                     f"{a['fold_sd']:.3f}; balanced accuracy {b['oof']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]; "
                     f"{r['seconds']:.0f} s.")
        for key, label in (("vs_row5", "row 5"), ("vs_best", f"best so far ({r.get('best_so_far')})")):
            c = r.get(key)
            if c:
                lines.append(f"- vs {label}: AUROC gain {c['gain']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], "
                             f"wins in {c['share_won']:.0%} of resamples.")
        if any(r["scores"]["chosen_per_fold"]):
            lines.append(f"- Chosen per fold: {r['scores']['chosen_per_fold']}")
        lines.append("")
    return "\n".join(lines)


def progress_plot(records: list[dict], row5: dict, winner: str | None, path: Path) -> None:
    exps = [r for r in records if r["type"] == "experiment"]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(exps) + 2), 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(row5["auroc"], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(0.005, row5["auroc"], f"row 5 ({row5['auroc']:.3f})", color=MUTED, va="bottom", ha="left",
            fontsize=9, transform=ax.get_yaxis_transform())
    for phase, (color, marker, label) in PHASE_STYLE.items():
        pts = [(i, e) for i, e in enumerate(exps) if e["phase"] == phase and e["status"] == "done"]
        if not pts:
            continue
        x = [i for i, _ in pts]
        y = [e["scores"]["macro_auroc"]["oof"] for _, e in pts]
        lo = [v - e["scores"]["macro_auroc"]["ci"][0] for v, (_, e) in zip(y, pts)]
        hi = [e["scores"]["macro_auroc"]["ci"][1] - v for v, (_, e) in zip(y, pts)]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=marker, ms=8, color=color, ecolor=color, elinewidth=2,
                    capsize=0, mec=SURFACE, mew=2, label=label, zorder=3)
    for i, e in enumerate(exps):
        if e["id"] == winner:
            v = e["scores"]["macro_auroc"]["oof"]
            ax.scatter([i], [v], s=260, facecolors="none", edgecolors=INK, linewidths=2, zorder=4)
            ax.annotate("selected", (i, v), xytext=(0, 14), textcoords="offset points", ha="center",
                        color=INK, fontsize=9)
    ax.set_xticks(range(len(exps)))
    long = any(":" in e["recipe"] for e in exps)
    ax.set_xticklabels([e["recipe"] + ("" if long else (f"\n{e['size']}" if e["size"] else "")) for e in exps],
                       fontsize=8, color=INK, rotation=60 if long else 0, ha="right" if long else "center")
    ax.set_ylabel("Macro AUROC (out-of-fold, 95% CI)", color=INK)
    ax.set_title(f"Fusion search, seed {records[0]['seed']}", color=INK, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def winners_curse(records: list[dict], load_P, y: np.ndarray, idx: np.ndarray) -> dict:
    """How much the best development AUROC is inflated by being selected (Tibshirani & Tibshirani, 2009).

    For each stored resample: take the experiment that scores best on that resample, and
    compare its resample score with its score on all patients. The mean difference estimates
    the optimism of picking the best of several experiments on the same patients.
    """
    exps = [r for r in records if r["type"] == "experiment" and r["status"] == "done"]
    P = [load_P(e["id"]) for e in exps]
    full = np.array([macro_auroc(y, p) for p in P])
    infl = []
    for i in idx:
        on_b = np.array([macro_auroc(y[i], p[i]) for p in P])
        k = int(on_b.argmax())
        infl.append(on_b[k] - full[k])
    return {"experiments": len(exps), "inflation_mean": float(np.mean(infl)),
            "inflation_ci": [float(np.quantile(infl, 0.025)), float(np.quantile(infl, 0.975))]}


def selected(records: list[dict]) -> dict | None:
    final = next((r for r in records if r["type"] == "decision" and r["rule"] == "final_selection"), None)
    if final is None:
        return None
    e = next(r for r in records if r.get("id") == final["action"]["winner"])
    out = {"winner": e["id"], "recipe": e["recipe"], "size": e["size"], "phase": e["phase"],
           "candidate": e.get("candidate"), "mode": e.get("mode"),
           "settings": e["settings"], "chosen_per_fold": e["scores"]["chosen_per_fold"],
           "macro_auroc": e["scores"]["macro_auroc"], "balanced_accuracy": e["scores"]["balanced_accuracy"],
           "decision": final["id"], "fingerprint": e["fingerprint"]}
    out["checksum"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


def write_all(run_dir: Path, records: list[dict], row5: dict, load_P, y: np.ndarray, idx: np.ndarray,
              final: bool = True) -> None:
    """Regenerate every report; `selected.yaml` (with the winner's-curse estimate) only when `final`."""
    summary(records).to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "decision_log.md").write_text(decision_log(records, row5))
    sel = selected(records)
    progress_plot(records, row5, sel["winner"] if sel else None, run_dir / "progress.png")
    if sel and final:
        sel["winners_curse"] = winners_curse(records, load_P, y, idx)
        (run_dir / "selected.yaml").write_text(yaml.safe_dump(sel, sort_keys=False))
