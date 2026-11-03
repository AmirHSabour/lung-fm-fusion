"""Stage 10: the fusion-search agent for one seed (docs/agent.md; decisions/vNN.md).

Runs the structured inner agent of one version: the policy named in the config proposes
experiments and decisions; every experiment and decision is appended to `trials.jsonl`; the
run can be stopped and restarted at any time and resumes from the record. Reports
(`summary.csv`, `decision_log.md`, `progress.png`, `selected.yaml`) are regenerated from the
record. Development patients only.

Usage (from the repo root):
    uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/v02.yaml
    uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/v02.yaml --matched
    uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dev --config conf/versions/v11.yaml \
        --out-dir artifacts/rerun/v11 --device cuda:1                                    # re-run a recorded version
    uv run --no-sync python -m scripts.10_fusion_search --seed 0 --dry-run --models uni2h virchow2   # v01
    uv run --no-sync python -m scripts.10_fusion_search --seed 0                                     # official (v01)
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")            # one BLAS thread per fit (docs/methods.md D21)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # device numbers as in nvidia-smi

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.agent.evaluate import compare, cross_validate, scores
from src.agent.policies import get_policy
from src.agent.policy import Decide, Done
from src.agent.report import write_all
from src.agent.store import TrialStore, code_version, fingerprint
from src.data.cohort import CLASSES
from src.embeddings.store import sha256_file
from src.eval.dataset import load_dev
from src.eval.metrics import balanced_accuracy, macro_auroc, stratified_resamples
from src.fusion.blocks import ID, META, Blocks
from src.fusion.registry import RECIPES, settings
from src.utils.config import load_config
from src.utils.gpu import gpu_name


GPU_FAMILIES = ("learned", "mil", "mexp", "agg", "clam", "ctx", "ens", "xq", "xqm", "hier")      # families that train networks on the GPU


def check_gpu(device: str, gpu_type: str) -> str:
    """The GPU's name; a note is printed if it is not the type the reported results used."""
    return gpu_name(device, gpu_type)


def check_official(cfg: dict, models: list[str], baselines: Path) -> None:
    if models != cfg["models"]:
        raise SystemExit(f"the official run uses all models {cfg['models']}; use --dry-run or --dev for {models}")
    if json.loads((baselines / "best_single_plus_metadata.json").read_text())["provisional"]:
        raise SystemExit("row 5 is still provisional; rerun stage 09 with all three models first")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "conf/agent.yaml"], capture_output=True)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "conf/agent.yaml"])
    if tracked.returncode or clean.returncode:
        raise SystemExit("conf/agent.yaml must be committed (frozen) before the official run")
    if code_version().endswith("-dirty"):
        raise SystemExit("commit all code changes first: the record names the exact code version it used")


def candidate_settings(candidate: str, mode: str, cfg: dict) -> dict:
    from src.fusion.candidates import family, split
    recipe, encoding = split(candidate)
    fam, p1 = family(candidate), cfg["phase1"]
    s = {"candidate": candidate, "recipe": recipe, "encoding": encoding, "family": fam, "mode": mode,
         **RECIPES[recipe].options}
    if mode == "screen":
        s |= {"early": {"C": p1["C_img"], "size": p1["size"] if RECIPES[recipe].has_size else None},
              "late": {"C_img": p1["C_img"], "C_meta": p1["C_meta"], "C_stack": p1["C_stack"]},
              "pool": {"C_img": p1["C_img"], "C_meta": p1["C_meta"]},
              "mil": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil")},
              "mexp": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                       "C_meta": p1["C_meta"]},
              "agg": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                      "C_meta": p1["C_meta"]},
              "clam": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                       "C_meta": p1["C_meta"]},
              "ctx": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                      "C_meta": p1["C_meta"], "ctx_cache": cfg.get("ctx_cache")},
              "ens": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                      "C_meta": p1["C_meta"], "ctx_cache": cfg.get("ctx_cache")},
              "xq": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                     "C_meta": p1["C_meta"]},
              "xqm": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil")},
              "hier": {"weight_decay": p1["weight_decay"], "d": p1["size"], "mil": cfg.get("mil"),
                       "C_meta": p1["C_meta"]},
              "learned": {"weight_decay": p1["weight_decay"], "d": p1["size"]}}[fam]
    else:
        s |= {"tuned_by_inner_cv": cfg["phase2"]["grid"], "inner_folds": cfg["phase2"]["inner_folds"],
              "size_grid": cfg["phase2"]["size_grid"] if (RECIPES[recipe].has_size) else None}
        if fam == "mil":
            s |= {"mil": cfg.get("mil"), "d": p1["size"]}
    if fam == "learned":
        s |= {"network": cfg["network"]}
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dev", action="store_true", help="development run of a version; outputs in the config's out_dir")
    ap.add_argument("--dry-run", action="store_true", help="v01 dry run: any subset of models; outputs under dryrun_dir")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--ablation", action="store_true", help="v01: compute row 5b from the finished search")
    ap.add_argument("--matched", action="store_true", help="v02+: matched baselines and metadata diagnostic")
    ap.add_argument("--config", type=Path, default=Path("conf/agent.yaml"))
    ap.add_argument("--out-dir", type=Path, help="--dev: output root instead of the config's out_dir (to re-run a "
                                                 "recorded version into a fresh directory)")
    ap.add_argument("--device", help="GPU for the networks instead of the config's device, e.g. cuda:1")
    args = ap.parse_args()
    cfg, train, data = load_config(args.config), load_config("conf/train.yaml"), load_config("conf/data.yaml")
    policy_name = cfg.get("policy", "v01")
    next_action = get_policy(policy_name)
    models = args.models or cfg["models"]
    baselines = Path(train["runs_dir"]) / f"seed_{args.seed}" / "baselines"
    if args.dev:
        run_dir = (args.out_dir or Path(cfg["out_dir"])) / f"seed_{args.seed}" / "agent"
        mode = "dev"
    elif args.dry_run:
        run_dir = Path(cfg["dryrun_dir"]) / f"seed_{args.seed}" / "agent"
        mode = "dry_run"
    else:
        check_official(cfg, models, baselines)
        run_dir = Path(cfg["runs_dir"]) / f"seed_{args.seed}" / "agent"
        mode = "official"
    (run_dir / "trials").mkdir(parents=True, exist_ok=True)
    cfg["expert_cache"] = str(run_dir / "work" / "experts")      # v05+: cached MIL-expert predictions (not a setting)
    device = cfg["device"] = args.device or cfg["device"]
    gpu = check_gpu(device, train["gpu_type"])

    # Data: development patients only; blocks side by side, age and sex last.
    feats = Path(train["features_dir"])
    dev = load_dev(args.seed, models, splits_dir=Path(data["splits_dir"]), cohort_csv=Path(data["cohort_csv"]),
                   features_dir=feats)
    X = np.hstack([dev.blocks[m] for m in models] + [dev.meta])
    blocks = Blocks.from_sizes({**{m: dev.blocks[m].shape[1] for m in models}, META: dev.meta.shape[1]})
    if cfg.get("tile_cache"):                   # v04+: each patient's row in the tile cache, read by MIL recipes only
        from src.embeddings.tile_cache import index as cache_index
        pos_c = cache_index(cfg["tile_cache"])
        ids = np.array([[pos_c[w]] for w in dev.wsi_ids], dtype=float)
        X = np.hstack([X[:, :-dev.meta.shape[1]], ids, dev.meta])
        blocks = Blocks.from_sizes({**{m: dev.blocks[m].shape[1] for m in models}, ID: 1, META: dev.meta.shape[1]})
    y = dev.y

    # Row 5, the reference (stage 09 output, aligned by WSI_ID).
    r5 = np.load(baselines / "best_single_plus_metadata.npz", allow_pickle=True)
    pos = {w: i for i, w in enumerate(r5["WSI_ID"])}
    P5 = r5["P"][[pos[w] for w in dev.wsi_ids]]
    row5_model = json.loads((baselines / "best_single_plus_metadata.json").read_text())["model"]
    row5 = {"auroc": macro_auroc(y, P5), "bacc": balanced_accuracy(y, P5), "model": row5_model}

    # The seed's paired resamples, drawn once and reused by every comparison.
    idx_path = run_dir / "bootstrap_idx.npy"
    if not idx_path.exists():
        np.save(idx_path, stratified_resamples(y, cfg["search"]["n_resamples"], args.seed))
    idx = np.load(idx_path)

    header = {"seed": args.seed, "mode": mode, "models": models,
              "code": code_version(), "features": {m: sha256_file(feats / f"{m}.npz") for m in models},
              "config": sha256_file(args.config)}
    store = TrialStore(run_dir / "trials.jsonl", args.seed)
    store.open_run(header)

    def load_P(trial_id: str) -> np.ndarray:
        return np.load(run_dir / "trials" / f"{trial_id}.npz")["P"]

    @lru_cache(maxsize=None)
    def compare_ids(a: str, b: str) -> dict:
        return compare(y, load_P(a), load_P(b), idx)

    def evaluate(spec: dict, Xr: np.ndarray, br: Blocks, rec: dict, family: str) -> dict:
        print(f"[{rec['id']}] {rec['recipe']} ({rec.get('mode', 'phase ' + str(rec.get('phase')))}): {rec['reason']}",
              flush=True)
        status, P, per_fold, sec = cross_validate(spec, Xr, y, dev.folds, br, cfg, args.seed, device,
                                                  run_dir / "work", cfg["search"]["cap_minutes"] * 60)
        rec |= {"status": status, "seconds": sec, "code": header["code"],
                "device": gpu if family in GPU_FAMILIES else "cpu"}
        if status == "done":
            np.savez(run_dir / "trials" / f"{rec['id']}.npz", P=P, y=y, WSI_ID=dev.wsi_ids)
            rec["scores"] = scores(y, P, per_fold, idx, CLASSES)
            rec["vs_row5"] = compare(y, P, P5, idx)
            prior = [r for r in store.records() if r["type"] == "experiment" and r["status"] == "done"]
            if prior:
                best = max(prior, key=lambda r: r["scores"]["macro_auroc"]["oof"])
                rec["best_so_far"] = best["id"]
                rec["vs_best"] = compare(y, P, load_P(best["id"]), idx)
            print(f"    AUROC {rec['scores']['macro_auroc']['oof']:.3f}, balanced accuracy "
                  f"{rec['scores']['balanced_accuracy']['oof']:.3f}, {sec:.0f} s", flush=True)
        else:
            print(f"    {status} after {sec:.0f} s", flush=True)
        return store.append(rec)

    # The agent loop.
    while not isinstance(action := next_action(store.records(), cfg, compare_ids), Done):
        if isinstance(action, Decide):
            rec = store.append({"type": "decision", "id": store.next_id("D"), "rule": action.rule,
                                "based_on": action.based_on, "statistics": action.statistics,
                                "action": action.action, "reason": action.reason, "policy": policy_name})
            print(f"[{rec['id']}] {action.rule}: {action.reason}", flush=True)
            continue
        if hasattr(action, "candidate"):            # v02+: recipe × encoding, screen or tuned
            from src.fusion.candidates import family
            sett = candidate_settings(action.candidate, action.mode, cfg)
            spec = {"candidate": action.candidate, "mode": action.mode}
            rec = {"type": "experiment", "id": store.next_id("T"), "seed": args.seed, "policy": policy_name,
                   "candidate": action.candidate, "mode": action.mode, "recipe": action.candidate,
                   "phase": 1 if action.mode == "screen" else 2, "encoding": sett["encoding"],
                   "size": sett.get("size") if action.mode == "screen" else None}
            fam = family(action.candidate)
        else:                                       # v01: recipe, phase, size
            sett = settings(action.recipe, action.phase, action.size, cfg)
            spec = {"recipe": action.recipe, "phase": action.phase, "size": action.size}
            rec = {"type": "experiment", "id": store.next_id("T"), "seed": args.seed, "phase": action.phase,
                   "recipe": action.recipe, "size": action.size}
            fam = RECIPES[action.recipe].family
        fp = fingerprint({**sett, "models": models, "features": header["features"]})
        if any(r.get("fingerprint") == fp for r in store.records()):
            raise RuntimeError(f"{action} is already recorded; the policy must not repeat an experiment")
        evaluate(spec, X, blocks, rec | {"settings": sett, "fingerprint": fp, "based_on": action.based_on,
                                         "reason": action.reason}, fam)
        write_all(run_dir, store.records(), row5, load_P, y, idx, final=False)
    print(action.reason, flush=True)

    final = next((r for r in store.of_type("decision") if r["rule"] == "final_selection"), None)

    def run_extra(kind: str, label: str, names: list[str], spec: dict, reason: str, fam: str) -> dict | None:
        if any(r.get("label") == label for r in store.of_type(kind)):
            return None
        if ID in blocks.names:
            names = [n for n in names if n != META] + [ID] + ([META] if META in names else [])
        sub = blocks.subset(names)
        return evaluate(spec, X[:, blocks.columns(names)], sub,
                        {"type": kind, "id": store.next_id("A"), "seed": args.seed, "label": label, "blocks": names,
                         "recipe": spec.get("candidate", spec.get("recipe")), "mode": spec.get("mode"),
                         "phase": spec.get("phase", 2 if spec.get("mode") == "tuned" else 1),
                         "size": spec.get("size"), "settings": spec,
                         "fingerprint": fingerprint({**spec, "blocks": names}), "based_on": [final["action"]["winner"]],
                         "reason": reason}, fam)

    # v01: row 5b, the winner's way of bringing in age and sex, applied once to row 5's model.
    if args.ablation and final and "winner_recipe" in final["action"] and not store.of_type("ablation"):
        a = final["action"]
        spec = {"recipe": a["winner_recipe"], "phase": a["winner_phase"], "size": a["winner_size"]}
        run_extra("ablation", "5b", [row5_model, META], spec,
                  f"Row 5b: the selected recipe's handling of age and sex, applied to row 5's model ({row5_model}) alone",
                  RECIPES[a["winner_recipe"]].family)

    # v02+: matched baselines (the winner's recipe, encoding and tuning on fewer sources) and the metadata diagnostic.
    if args.matched and final and "winner_candidate" in final["action"]:
        from src.fusion.candidates import family
        a = final["action"]
        spec = {"candidate": a["winner_candidate"], "mode": a["winner_mode"]}
        fam = family(a["winner_candidate"])
        rows = [("1m", [META], "age and sex alone")] + [(f"{m}_m", [m], f"{m} alone") for m in models] + \
               [("diag_no_meta", list(models), "all image models without age and sex (metadata diagnostic)")]
        for label, names, what in rows:
            if fam == "learned" and spec["candidate"].startswith("N3") and (names == [META] or META not in names):
                continue      # the bilinear mixer needs both image streams and age and sex
            if fam == "mil" and (names == [META] or (spec["candidate"].startswith("M3") and META not in names)):
                continue      # MIL needs tiles; FiLM needs age and sex
            if fam == "xqm" and names == [META]:
                continue      # v10: Q2's experts need tiles (age and sex enter them as queries)
            run_extra("matched", label, names, spec, f"Matched baseline {label}: winner {spec['candidate']} "
                      f"({spec['mode']}) on {what}", fam)
        singles = [r for r in store.of_type("matched") if r["label"].endswith("_m") and r["status"] == "done"]
        if singles:
            best_single = max(singles, key=lambda r: r["scores"]["macro_auroc"]["oof"])["blocks"][0]
            run_extra("matched", "5m", [best_single, META], spec,
                      f"Matched row 5: winner {spec['candidate']} ({spec['mode']}) on the best matched single model "
                      f"({best_single}) plus age and sex", fam)
    write_all(run_dir, store.records(), row5, load_P, y, idx, final=True)
    print(f"-> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
