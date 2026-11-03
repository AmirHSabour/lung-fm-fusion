"""v02 policy: open one candidate per family, follow the leading family, drop families that are
clearly behind, tune the best three, select (decisions/v02.md §2.3).

A pure function of the records: `next_action` returns the next experiment (`ProposeCandidate`),
a decision to record (`Decide`), or `Done`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.policy import Compare, Decide, Done
from src.fusion.candidates import all_candidates, families, family, split


@dataclass(frozen=True)
class ProposeCandidate:
    candidate: str
    mode: str                    # screen | tuned
    based_on: list[str] = field(default_factory=list)
    reason: str = ""


def _auroc(e: dict) -> float:
    return e["scores"]["macro_auroc"]["oof"]


def _bacc(e: dict) -> float:
    return e["scores"]["balanced_accuracy"]["oof"]


def _fmt(e: dict) -> str:
    return f"{e['id']} ({e['candidate']} {e['mode']}, AUROC {_auroc(e):.3f})"


def simplicity_key(e: dict, cands: list[str]) -> tuple:
    return (cands.index(e["candidate"]), e["mode"] == "tuned")


def next_action(records: list[dict], cfg: dict, compare: Compare):
    s = cfg["search"]
    cands = all_candidates(cfg)
    exps = [r for r in records if r["type"] == "experiment"]
    done = [e for e in exps if e["status"] == "done"]
    decs = [r for r in records if r["type"] == "decision"]
    by_rule = lambda rule: [d for d in decs if d["rule"] == rule]

    # 1. Open: one screen per family.
    if not by_rule("open"):
        opening = cfg["opening"]
        what = ("one candidate per family" if len({family(c) for c in opening}) == len(opening)
                else "the simplest candidate of each family plus every recipe of the families opened in full")
        return Decide("open", [], {}, {"candidates": opening}, f"Open with {what} at fixed values: {', '.join(opening)}.")
    open_id = by_rule("open")[0]["id"]
    tried_screen = {e["candidate"] for e in exps if e["mode"] == "screen"}
    for c in by_rule("open")[0]["action"]["candidates"]:
        if c not in tried_screen:
            return ProposeCandidate(c, "screen", [open_id], f"Opening: {family(c)} family, simplest candidate")

    if by_rule("final_selection"):
        return Done(f"final selection {by_rule('final_selection')[0]['id']}: "
                    f"{by_rule('final_selection')[0]['action']['winner_candidate']}")

    dropped = {d["action"]["family"] for d in by_rule("drop")}
    fams = families(cfg)
    in_play = [f for f in fams if f not in dropped]
    fam_done = {f: [e for e in done if family(e["candidate"]) == f] for f in fams}
    best = max(done, key=_auroc) if done else None

    if not by_rule("tune_start"):
        # 2a. Drop families that are clearly behind (wins >= win_share of paired resamples).
        for f in in_play:
            if len([e for e in exps if family(e["candidate"]) == f]) < 2 or not fam_done[f]:
                continue
            fb = max(fam_done[f], key=_auroc)
            if family(best["candidate"]) == f:
                continue
            c = compare(best["id"], fb["id"])
            if c["share_won"] >= s["win_share"]:
                return Decide("drop", [best["id"], fb["id"]], {"best_wins_share": c["share_won"], "gain": c["gain"]},
                              {"family": f},
                              f"Drop {f}: the best experiment {_fmt(best)} beats the {f} family's best {_fmt(fb)} "
                              f"in {c['share_won']:.0%} of resamples (≥ {s['win_share']:.0%}).")
        # 2b. Explore within the leading family.
        n_tune = s["tune_count"]
        untried = {f: [c for c in cands if family(c) == f and c not in tried_screen] for f in in_play}
        live = [f for f in in_play if untried[f] and fam_done[f]]
        if len(exps) < s["budget"] - n_tune and live:
            leader = max(live, key=lambda f: (max(_auroc(e) for e in fam_done[f]), -fams.index(f)))
            last = by_rule("leader")
            if not last or last[-1]["action"]["family"] != leader:
                lb = max(fam_done[leader], key=_auroc)
                return Decide("leader", [lb["id"]], {f: max(_auroc(e) for e in fam_done[f]) for f in live},
                              {"family": leader},
                              f"Leading family: {leader} (best {_fmt(lb)}); explore it next.")
            nxt = _next_in_family(leader, untried[leader], fam_done[leader], cfg)
            lb = max(fam_done[leader], key=_auroc)
            return ProposeCandidate(nxt, "screen", [last[-1]["id"], lb["id"]],
                                    f"Explore the leading family {leader}: {_why_next(nxt, fam_done[leader], cfg)}")
        # 3. Start tuning: best screened candidate of each recipe, top `tune_count`, families in play.
        untunable = set(cfg.get("untunable_families", []))
        screened = [e for e in done if e["mode"] == "screen" and family(e["candidate"]) in in_play
                    and family(e["candidate"]) not in untunable]
        best_per_recipe = {}
        for e in sorted(screened, key=_auroc, reverse=True):
            best_per_recipe.setdefault(split(e["candidate"])[0], e)
        top = sorted(best_per_recipe.values(), key=_auroc, reverse=True)[:n_tune]
        why = "the exploration budget is used" if len(exps) >= s["budget"] - n_tune else "no untried candidate is left"
        return Decide("tune_start", [e["id"] for e in top], {e["id"]: _auroc(e) for e in top},
                      {"tune": [e["candidate"] for e in top]},
                      f"Start tuning ({why}): the best screened candidate of each recipe, top {n_tune}: "
                      + ", ".join(_fmt(e) for e in top) + ".")

    # 4. Tune.
    ts = by_rule("tune_start")[0]
    tuned = {e["candidate"] for e in exps if e["mode"] == "tuned"}
    for c in ts["action"]["tune"]:
        if c not in tuned and len(exps) < s["budget"]:
            return ProposeCandidate(c, "tuned", [ts["id"]], f"Tune {c}: C or weight decay and size by inner CV")

    # 5. Select among all finished experiments of the families in play.
    pool = [e for e in done if family(e["candidate"]) in in_play]
    cand = max(pool, key=_auroc)
    stats, winner = {}, cand
    for e in sorted(pool, key=lambda x: simplicity_key(x, cands)):
        if e is cand:
            break
        c = compare(cand["id"], e["id"])
        clear = c["share_won"] >= s["win_share"] and _bacc(cand) >= _bacc(e) - s["bacc_margin"]
        stats[e["id"]] = {"candidate": e["candidate"], "mode": e["mode"], "candidate_wins_share": c["share_won"],
                          "gain": c["gain"], "bacc_difference": _bacc(cand) - _bacc(e), "clearly_beaten": clear}
        if not clear:
            winner = e
            break
    if winner is cand:
        why = f"{_fmt(cand)} has the highest AUROC and clearly beats every simpler experiment of the families in play."
    else:
        st = stats[winner["id"]]
        why = (f"{_fmt(cand)} has the highest AUROC but does not clearly beat the simpler {_fmt(winner)}: it wins in "
               f"{st['candidate_wins_share']:.0%} of resamples (needs ≥ {s['win_share']:.0%}) and its balanced accuracy "
               f"differs by {st['bacc_difference']:+.3f} (may be at most {s['bacc_margin']} lower); the simpler one is chosen.")
    return Decide("final_selection", [e["id"] for e in pool], stats,
                  {"winner": winner["id"], "winner_candidate": winner["candidate"], "winner_mode": winner["mode"],
                   "candidate": cand["id"]}, why)


def _next_in_family(fam: str, untried: list[str], fam_done: list[dict], cfg: dict) -> str:
    """Untried recipes at the default encoding first; then untried encodings of the best recipes."""
    default = cfg["default_encoding"][fam]
    first = [c for c in untried if split(c)[1] == default]
    if first:
        return first[0]
    recipe_best = {}
    for e in sorted(fam_done, key=_auroc, reverse=True):
        recipe_best.setdefault(split(e["candidate"])[0], _auroc(e))
    for recipe in sorted(recipe_best, key=recipe_best.get, reverse=True):
        for c in untried:
            if split(c)[0] == recipe:
                return c
    return untried[0]


def _why_next(c: str, fam_done: list[dict], cfg: dict) -> str:
    recipe, enc = split(c)
    if enc == cfg["default_encoding"][family(c)]:
        return f"untried recipe {recipe} at the default encoding {enc}"
    return f"untried age-and-sex encoding {enc} for {recipe}, the family's best recipe without it"
