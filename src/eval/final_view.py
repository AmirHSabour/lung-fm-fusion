"""Tables and figures of the final evaluation and the version loop, for the notebooks.

Everything is read from saved records; nothing here trains a model:
- `artifacts/final/seed_<s>/results.json` (stage 12): test and development scores of every
  design and row, paired fused-vs-row comparisons, per-class AUROC, confusion matrices;
- `artifacts/versions/leaderboard_seed0.csv` and `artifacts/versions/vNN/seed_0/agent/` (stage 10):
  the version loop.

Figures follow one style: light surface, three categorical colours at most per chart, thin marks,
one y-axis, legends for two or more series.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.cohort import CLASSES

# Palette (validated light-mode reference palette: categorical slots 1-3, ink, grid).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ROW_NAMES = {"fused": "Fused + age and sex", "no_meta": "Fused without age and sex",
             "uni2h": "UNI2-h alone", "virchow2": "Virchow2 alone", "genbio_pathfm": "GenBio-PathFM alone",
             "best_plus_meta": "Best single model + age and sex", "meta": "Age and sex only"}
BASE_NAMES = {"metadata": "Age and sex only", "uni2h": "UNI2-h alone", "virchow2": "Virchow2 alone",
              "genbio_pathfm": "GenBio-PathFM alone", "best_single_plus_metadata": "Best single model + age and sex"}
TABLE_ROWS = ["meta", "uni2h", "virchow2", "genbio_pathfm", "best_plus_meta", "fused"]


def style() -> None:
    """Matplotlib defaults for every figure in the notebooks."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 10, "figure.dpi": 110})


# ---- loading ------------------------------------------------------------------------------

def load_final(root: str | Path = "artifacts/final") -> dict[int, dict]:
    """{seed: results.json} for every finished seed under `root`."""
    runs = {int(p.parent.name.split("_")[1]): json.loads(p.read_text())
            for p in sorted(Path(root).glob("seed_*/results.json"))}
    if not runs:
        raise FileNotFoundError(f"no seed_*/results.json under {root}")
    return runs


def designs(runs: dict[int, dict]) -> list[str]:
    return list(next(iter(runs.values()))["designs"])


def primary(runs: dict[int, dict]) -> str:
    first = next(iter(runs.values()))["designs"]
    return next(d for d, v in first.items() if v["role"] == "primary")


# ---- tables -------------------------------------------------------------------------------

def _ms(v: list[float]) -> str:
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def task_table(runs: dict[int, dict], design: str) -> pd.DataFrame:
    """TASK.md §5.3 table for one design: every row under the design's own recipe, test set,
    mean ± sd over seeds, plus seed 0 with its 95% bootstrap CI."""
    seeds = sorted(runs)
    out = []
    for row in TABLE_ROWS:
        r = {s: runs[s]["designs"][design]["rows"][row] for s in seeds}
        auc = [r[s]["test"]["macro_auroc"]["estimate"] for s in seeds]
        bac = [r[s]["test"]["balanced_accuracy"]["estimate"] for s in seeds]
        rec = {"Configuration": ROW_NAMES[row], "Test macro AUROC (mean ± sd)": _ms(auc),
               "Test balanced accuracy (mean ± sd)": _ms(bac)}
        if 0 in r:
            a, b = r[0]["test"]["macro_auroc"], r[0]["test"]["balanced_accuracy"]
            rec["Seed 0 AUROC [95% CI]"] = f"{a['estimate']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
            rec["Seed 0 bal. acc. [95% CI]"] = f"{b['estimate']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]"
        out.append(rec)
    return pd.DataFrame(out).set_index("Configuration")


def baseline_table(runs: dict[int, dict]) -> pd.DataFrame:
    """The original rows 1-5 (mean-pooled slide vectors + tuned logistic regression), test set."""
    seeds = sorted(runs)
    out = []
    for row, name in BASE_NAMES.items():
        b = {s: runs[s]["baselines"][row] for s in seeds}
        out.append({"Configuration": name,
                    "Test macro AUROC (mean ± sd)": _ms([b[s]["test"]["macro_auroc"]["estimate"] for s in seeds]),
                    "Test balanced accuracy (mean ± sd)": _ms([b[s]["test"]["balanced_accuracy"]["estimate"]
                                                              for s in seeds])})
    return pd.DataFrame(out).set_index("Configuration")


def gain_table(runs: dict[int, dict], design: str) -> pd.DataFrame:
    """Fused minus each row on the test set: gain per seed, mean ± sd, and on how many seeds the
    paired 95% CI lies above 0 (the claim rule fixed in conf/final.yaml)."""
    seeds = sorted(runs)
    out = []
    comps = [("same recipe", r, ROW_NAMES[r], "fused_vs_row_on_test")
             for r in ["uni2h", "virchow2", "genbio_pathfm", "best_plus_meta", "no_meta", "meta"]]
    comps += [("original rows 1–5", r, BASE_NAMES[r], "fused_vs_baseline_on_test") for r in BASE_NAMES]
    for group, key, name, field in comps:
        cs = [runs[s]["designs"][design][field][key] for s in seeds]
        g = [c["gain"] for c in cs]
        out.append({"Compared with": f"{name} ({group})",
                    **{f"Seed {s}": f"{c['gain']:+.3f}" for s, c in zip(seeds, cs)},
                    "Mean gain ± sd": f"{np.mean(g):+.3f} ± {np.std(g, ddof=1):.3f}",
                    "Seeds with gain > 0": f"{sum(x > 0 for x in g)}/{len(g)}",
                    "Seeds with CI > 0": f"{sum(c['ci'][0] > 0 for c in cs)}/{len(cs)}"})
    return pd.DataFrame(out).set_index("Compared with")


def designs_table(runs: dict[int, dict]) -> pd.DataFrame:
    """The three frozen designs side by side: fused test and development AUROC."""
    seeds = sorted(runs)
    out = []
    for d in designs(runs):
        r = {s: runs[s]["designs"][d] for s in seeds}
        f = {s: r[s]["rows"]["fused"] for s in seeds}
        out.append({"Design": d, "Candidate": r[seeds[0]]["candidate"], "Role": r[seeds[0]]["role"],
                    "Test AUROC (mean ± sd)": _ms([f[s]["test"]["macro_auroc"]["estimate"] for s in seeds]),
                    "Test bal. acc. (mean ± sd)": _ms([f[s]["test"]["balanced_accuracy"]["estimate"] for s in seeds]),
                    "Development AUROC (mean ± sd)": _ms([f[s]["dev"]["macro_auroc"]["estimate"] for s in seeds])})
    return pd.DataFrame(out).set_index("Design")


def compare_runs(a: dict[int, dict], b: dict[int, dict]) -> pd.DataFrame:
    """Largest absolute difference between two runs of stage 12, per seed (0 = identical)."""
    out = []
    for s in sorted(set(a) & set(b)):
        diffs = []
        for d in a[s]["designs"]:
            for row, r in a[s]["designs"][d]["rows"].items():
                for part in ("dev", "test"):
                    for m in ("macro_auroc", "balanced_accuracy"):
                        diffs.append(abs(r[part][m]["estimate"] - b[s]["designs"][d]["rows"][row][part][m]["estimate"]))
        for row, r in a[s]["baselines"].items():
            for m in ("macro_auroc", "balanced_accuracy"):
                diffs.append(abs(r["test"][m]["estimate"] - b[s]["baselines"][row]["test"][m]["estimate"]))
        out.append({"Seed": s, "Numbers compared": len(diffs), "Largest difference": max(diffs)})
    return pd.DataFrame(out).set_index("Seed")


# ---- figures ------------------------------------------------------------------------------

def plot_per_seed(runs: dict[int, dict], design: str) -> plt.Figure:
    """Test macro AUROC per seed with 95% CIs: fused, the best single model under the same
    recipe (UNI2-h alone) and the original row 5."""
    seeds = sorted(runs)
    series = [("Fused + age and sex", BLUE, lambda s: runs[s]["designs"][design]["rows"]["fused"]["test"]),
              ("UNI2-h alone (same recipe)", ORANGE, lambda s: runs[s]["designs"][design]["rows"]["uni2h"]["test"]),
              ("Original row 5 (mean pooling + logistic)", AQUA,
               lambda s: runs[s]["baselines"]["best_single_plus_metadata"]["test"])]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (name, colour, get) in enumerate(series):
        x = np.arange(len(seeds)) + (k - 1) * 0.22
        est = [get(s)["macro_auroc"]["estimate"] for s in seeds]
        lo = [get(s)["macro_auroc"]["ci"][0] for s in seeds]
        hi = [get(s)["macro_auroc"]["ci"][1] for s in seeds]
        ax.errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o", ms=7, color=colour,
                    ecolor=colour, elinewidth=1.5, capsize=0, label=name, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xticks(range(len(seeds)), [f"Seed {s}" for s in seeds])
    ax.set_ylabel("Test macro AUROC (95% CI)")
    ax.set_title(f"{design}: test AUROC on each of the {len(seeds)} test sets")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.tight_layout()
    return fig


def plot_gains(runs: dict[int, dict], design: str) -> plt.Figure:
    """Fused minus each comparison row on the test set: one dot per seed, the mean as a bar."""
    seeds = sorted(runs)
    rows = [("UNI2-h alone", "fused_vs_row_on_test", "uni2h"),
            ("Virchow2 alone", "fused_vs_row_on_test", "virchow2"),
            ("GenBio-PathFM alone", "fused_vs_row_on_test", "genbio_pathfm"),
            ("Best single + age and sex", "fused_vs_row_on_test", "best_plus_meta"),
            ("Fused without age and sex", "fused_vs_row_on_test", "no_meta"),
            ("Original row 5\n(mean pooling + logistic)", "fused_vs_baseline_on_test", "best_single_plus_metadata")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (name, field, key) in enumerate(rows):
        g = [runs[s]["designs"][design][field][key]["gain"] for s in seeds]
        ax.barh(i, np.mean(g), height=0.55, color=BLUE if i < 5 else AQUA, alpha=0.35)
        ax.scatter(g, [i] * len(g), s=28, color=BLUE if i < 5 else AQUA, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.text(max(max(g), np.mean(g)) + 0.004, i, f"{np.mean(g):+.3f}", va="center", color=INK_2, fontsize=9)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Test AUROC gain of the fused model (dots: seeds; bar: mean)")
    ax.set_title(f"{design}: how much the fused model gains over each row")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_per_class(runs: dict[int, dict], design: str) -> plt.Figure:
    """Per-class test AUROC, mean over seeds: fused, UNI2-h alone (same recipe), original row 5."""
    seeds = sorted(runs)
    series = [("Fused + age and sex", BLUE, lambda s: runs[s]["designs"][design]["rows"]["fused"]["test"]),
              ("UNI2-h alone (same recipe)", ORANGE, lambda s: runs[s]["designs"][design]["rows"]["uni2h"]["test"]),
              ("Original row 5", AQUA, lambda s: runs[s]["baselines"]["best_single_plus_metadata"]["test"])]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(CLASSES))
    for k, (name, colour, get) in enumerate(series):
        v = [np.mean([get(s)["per_class_auroc"][c] for s in seeds]) for c in CLASSES]
        ax.bar(x + (k - 1) * 0.27, v, width=0.25, color=colour, label=name, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test AUROC (mean over seeds)")
    ax.set_title(f"{design}: per-class test AUROC (3–9 test patients per class per seed)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion(runs: dict[int, dict], design: str) -> plt.Figure:
    """Test confusion matrix of the fused model, summed over seeds; colour = share of the true class."""
    seeds = sorted(runs)
    cm = np.sum([np.array(runs[s]["designs"][design]["rows"]["fused"]["test"]["confusion"]) for s in seeds], axis=0)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), [f"{c} ({cm[i].sum()})" for i, c in enumerate(CLASSES)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (number of test patients)")
    ax.grid(False)
    ax.set_title(f"{design} fused: test confusion matrix, seeds {seeds[0]}–{seeds[-1]} summed")
    fig.tight_layout()
    return fig


# ---- the version loop ---------------------------------------------------------------------

VERSION_IDEAS = {
    "v01": "Fixed sweep of 11 recipes (early, late, learned)", "v02": "Adaptive search; learned age-and-sex tokens",
    "v03": "Age and sex as a Bayesian prior", "v04": "Tile-level attention MIL",
    "v05": "One MIL expert per model × age-and-sex prior", "v06": "Composition pooling (+ GenBio-PathFM)",
    "v07": "CLAM experts", "v08": "Spatial context tokens", "v09": "Ensembles of experts",
    "v10": "Age and sex as attention queries", "v11": "Auxiliary grade-group target",
    "v01_uvg": "Re-baseline of v01 on three models", "v04_uvg": "Re-baseline of v04 on three models"}


def versions_table(root: str | Path = "artifacts/versions") -> pd.DataFrame:
    """One row per version: idea, models, the agent's winner and its development AUROC, the best
    experiment of the version, gain over row 5, winner's curse."""
    lb = pd.read_csv(Path(root) / "leaderboard_seed0.csv")
    out = []
    for r in lb.itertuples():
        summ = pd.read_csv(Path(root) / r.version / "seed_0" / "agent" / "summary.csv")
        ex = summ[(summ["type"] == "experiment") & (summ["status"] == "done")]
        best = ex.loc[ex["macro_auroc"].idxmax()]
        out.append({"Version": r.version, "Idea": VERSION_IDEAS.get(r.version, ""),
                    "Models": "U, V" if r.models == "uni2h+virchow2" else "U, V, G",
                    "Experiments": r.experiments, "Agent's winner": r.candidate, "Winner AUROC": r.auroc,
                    "Best experiment": best["recipe"], "Best AUROC": round(float(best["macro_auroc"]), 4),
                    "Gain vs row 5": r.gain_vs_row5, "Current best": r.current_best,
                    "Winner's curse": r.global_winners_curse})
    order = sorted(out, key=lambda d: (d["Version"][:3], d["Version"]))
    return pd.DataFrame(order).set_index("Version")


def plot_versions(table: pd.DataFrame, row5: float = 0.799) -> plt.Figure:
    """Development AUROC (seed 0) of each version: the agent's winner and the version's best experiment."""
    t = table[~table.index.str.endswith("_uvg")]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(x, t["Best AUROC"], "-o", color=ORANGE, linewidth=2, ms=6, label="Best experiment of the version")
    ax.plot(x, t["Winner AUROC"], "-o", color=BLUE, linewidth=2, ms=6, label="Agent's winner (selection rule)")
    ax.axhline(row5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(len(t) - 0.6, row5 + 0.003, f"Row 5 = {row5:.3f}", color=INK_2, fontsize=9, ha="right")
    ax.axvline(t.index.get_loc("v06") - 0.5, color=GRID, linewidth=1)
    ax.text(t.index.get_loc("v06") - 0.4, 0.840, "from v06:\nthree models", color=INK_2, fontsize=9, va="top")
    ax.set_xticks(x, t.index)
    ax.set_ylabel("Development macro AUROC (seed 0)")
    ax.set_title("The version loop: what each version reached on development data")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    return fig


def decisions(version: str, root: str | Path = "artifacts/versions") -> pd.DataFrame:
    """The decisions of one version's agent run, in order: rule, reason."""
    recs = [json.loads(line) for line in (Path(root) / version / "seed_0" / "agent" / "trials.jsonl").read_text()
            .splitlines() if line.strip()]
    return pd.DataFrame([{"ID": r["id"], "Rule": r["rule"], "Reason": r["reason"]}
                         for r in recs if r["type"] == "decision"]).set_index("ID")


def experiments(version: str, root: str | Path = "artifacts/versions") -> pd.DataFrame:
    """The experiments of one version's agent run, in order."""
    s = pd.read_csv(Path(root) / version / "seed_0" / "agent" / "summary.csv")
    s = s[s["type"] == "experiment"]
    return s[["id", "recipe", "mode", "status", "macro_auroc", "balanced_accuracy", "gain_vs_row5", "seconds"]] \
        .rename(columns={"id": "ID", "recipe": "Candidate", "mode": "Mode", "status": "Status",
                         "macro_auroc": "AUROC", "balanced_accuracy": "Bal. acc.", "gain_vs_row5": "vs row 5",
                         "seconds": "Seconds"}).set_index("ID").round(3)


# ---- re-running the final evaluation ------------------------------------------------------

def run_final_eval(seeds: list[int], devices: tuple[str, ...] = ("cuda:0", "cuda:1"),
                   out_dir: str = "artifacts/final_rerun", log_dir: str = "artifacts/logs") -> None:
    """Re-run stage 12 for `seeds`, one seed per GPU at a time (a 24 GB RTX 4090 holds one seed at
    peak), writing to `out_dir` so the committed results in artifacts/final are never overwritten.
    Then build the tables for the re-run (`<out_dir>/results.md`)."""
    import subprocess
    import sys
    import time
    queue, running = list(seeds), {}
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while queue or running:
        for dev in devices:
            if dev not in running and queue:
                s = queue.pop(0)
                log = open(Path(log_dir) / f"final_rerun_s{s}.log", "w")
                running[dev] = (s, subprocess.Popen([sys.executable, "-m", "scripts.12_final_eval", "--seed", str(s),
                                                     "--device", dev, "--out-dir", out_dir], stdout=log,
                                                    stderr=subprocess.STDOUT), log)
                print(f"[{(time.time() - t0) / 60:5.1f} min] seed {s} started on {dev}", flush=True)
        for dev, (s, proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[dev]
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode}; see {log.name})"
                print(f"[{(time.time() - t0) / 60:5.1f} min] seed {s} {status}", flush=True)
        time.sleep(5)
    subprocess.run([sys.executable, "-m", "scripts.13_final_report", "--dir", out_dir,
                    "--out", str(Path(out_dir) / "results.md")], check=True)
