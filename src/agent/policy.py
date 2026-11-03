"""The agent's decision rules (docs/agent.md).

`next_action` reads the records so far and returns the next step: an experiment to run, a
decision to record, or Done. It holds no other state, so a run can be stopped and resumed
at any point and replays the same way. Statistical comparisons between two experiments are
supplied by `compare(id_a, id_b)` (paired bootstrap on the seed's stored resamples).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.fusion.registry import RECIPES


@dataclass(frozen=True)
class Propose:
    recipe: str
    phase: int
    size: int | None
    based_on: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class Decide:
    rule: str
    based_on: list[str]
    statistics: dict
    action: dict
    reason: str


@dataclass(frozen=True)
class Done:
    reason: str


Compare = Callable[[str, str], dict]   # (a, b) -> {"gain", "ci", "share_won"}: how reliably a beats b


def _auroc(e: dict) -> float:
    return e["scores"]["macro_auroc"]["oof"]


def _bacc(e: dict) -> float:
    return e["scores"]["balanced_accuracy"]["oof"]


def _fmt(e: dict) -> str:
    size = f", size {e['size']}" if e["size"] else ""
    return f"{e['id']} ({e['recipe']}{size}, AUROC {_auroc(e):.3f})"


def next_action(records: list[dict], cfg: dict, compare: Compare) -> Propose | Decide | Done:
    exps = [r for r in records if r["type"] == "experiment"]
    decisions = [r for r in records if r["type"] == "decision"]
    order, s = cfg["order"], cfg["search"]
    p1_size = cfg["phase1"]["size"]

    # Phase 1: every recipe once, fixed values, fixed order.
    done_p1 = {e["recipe"] for e in exps if e["phase"] == 1}
    for i, name in enumerate(order):
        if name not in done_p1:
            return Propose(name, 1, p1_size if RECIPES[name].has_size else None, [],
                           f"Phase 1 screening, recipe {i + 1} of {len(order)}, fixed values, fixed order")

    # Shortlist.
    shortlist = next((d for d in decisions if d["rule"] == "shortlist"), None)
    if shortlist is None:
        return _shortlist([e for e in exps if e["phase"] == 1], s, compare)

    final = next((d for d in decisions if d["rule"] == "final_selection"), None)
    if final is not None:
        return Done(f"final selection {final['id']}: {final['action']['winner_recipe']}")

    # Phase 2: tune each shortlisted recipe; adaptive size path.
    if len(exps) < s["budget"]:
        for name in shortlist["action"]["shortlist"]:
            prop = _phase2_next(name, [e for e in exps if e["phase"] == 2 and e["recipe"] == name],
                                cfg, shortlist["id"])
            if prop is not None:
                return prop
    return _final([e for e in exps if e["phase"] == 2], shortlist, exps, cfg, compare)


def _shortlist(p1: list[dict], s: dict, compare: Compare) -> Decide:
    ok = sorted([e for e in p1 if e["status"] == "done"], key=_auroc, reverse=True)
    best = ok[0]
    stats, tied = {}, []
    for e in ok[1:]:
        c = compare(best["id"], e["id"])
        stats[e["id"]] = {"recipe": e["recipe"], "best_wins_share": c["share_won"], "gain": c["gain"]}
        if c["share_won"] < s["win_share"]:
            tied.append(e)
    chosen = [best] + tied[: s["shortlist_max"] - 1]
    excluded = [e["id"] for e in p1 if e["status"] != "done"]
    reason = (f"Best in Phase 1: {_fmt(best)}. A recipe stays in if the best does not win in at least "
              f"{s['win_share']:.0%} of paired resamples; {len(tied)} tied, keeping at most {s['shortlist_max']}: "
              + ", ".join(_fmt(e) for e in chosen) + "."
              + (f" Not eligible (timed out or failed): {excluded}." if excluded else ""))
    return Decide("shortlist", [e["id"] for e in p1], stats,
                  {"shortlist": [e["recipe"] for e in chosen], "phase1_ids": [e["id"] for e in chosen]}, reason)


def _phase2_next(name: str, mine: list[dict], cfg: dict, shortlist_id: str) -> Propose | None:
    if not RECIPES[name].has_size:
        if mine:
            return None
        return Propose(name, 2, None, [shortlist_id], f"Phase 2: tune {name} (C by inner CV); it has no size")
    first, up, down = cfg["phase2"]["sizes"]           # 32, 64, 16
    by_size = {e["size"]: e for e in mine}
    if first not in by_size:
        return Propose(name, 2, first, [shortlist_id], f"Phase 2: tune {name} at size {first}")
    if up not in by_size:
        return Propose(name, 2, up, [shortlist_id, by_size[first]["id"]],
                       f"Phase 2: {name} at size {up}, to see whether a larger size helps")
    a, b = by_size[first], by_size[up]
    larger_better = b["status"] == "done" and (a["status"] != "done" or _auroc(b) > _auroc(a))
    if not larger_better and down not in by_size:
        return Propose(name, 2, down, [shortlist_id, a["id"], b["id"]],
                       f"Phase 2: size {up} was not better than {first} for {name}; trying {down}")
    return None


def _final(p2: list[dict], shortlist: dict, exps: list[dict], cfg: dict, compare: Compare) -> Decide:
    s, order = cfg["search"], cfg["order"]
    best_of = {}
    for e in p2:
        if e["status"] == "done" and (e["recipe"] not in best_of or _auroc(e) > _auroc(best_of[e["recipe"]])):
            best_of[e["recipe"]] = e
    note = "" if len(exps) < s["budget"] else f" Budget of {s['budget']} experiments reached."
    if not best_of:   # nothing tuned finished: fall back to the best Phase 1 recipe
        e = next(x for x in exps if x["id"] == shortlist["action"]["phase1_ids"][0])
        return Decide("final_selection", [shortlist["id"]], {},
                      {"winner": e["id"], "winner_recipe": e["recipe"], "winner_size": e["size"],
                       "winner_phase": 1}, "No Phase 2 experiment finished; the best Phase 1 recipe is kept." + note)
    cand = max(best_of.values(), key=_auroc)
    stats = {}
    winner = cand
    for e in sorted(best_of.values(), key=lambda x: order.index(x["recipe"])):   # simplest first
        if e is cand:
            break
        c = compare(cand["id"], e["id"])
        clear = c["share_won"] >= s["win_share"] and _bacc(cand) >= _bacc(e) - s["bacc_margin"]
        stats[e["id"]] = {"recipe": e["recipe"], "candidate_wins_share": c["share_won"], "gain": c["gain"],
                          "bacc_difference": _bacc(cand) - _bacc(e), "clearly_beaten": clear}
        if not clear:
            winner = e
            break
    if winner is cand:
        why = (f"{_fmt(cand)} has the highest AUROC and clearly beats every simpler shortlisted recipe "
               f"(wins in ≥ {s['win_share']:.0%} of resamples, balanced accuracy at most {s['bacc_margin']} lower).")
    else:
        st = stats[winner["id"]]
        why = (f"{_fmt(cand)} has the highest AUROC but does not clearly beat the simpler {_fmt(winner)}: "
               f"it wins in {st['candidate_wins_share']:.0%} of resamples (needs ≥ {s['win_share']:.0%}) and its "
               f"balanced accuracy differs by {st['bacc_difference']:+.3f} (may be at most {s['bacc_margin']} lower); "
               "the simpler recipe is chosen.")
    return Decide("final_selection", [e["id"] for e in best_of.values()] + [shortlist["id"]], stats,
                  {"winner": winner["id"], "winner_recipe": winner["recipe"], "winner_size": winner["size"],
                   "winner_phase": 2, "candidate": cand["id"]}, why + note)
