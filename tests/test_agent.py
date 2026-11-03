"""The agent's rules and record keeping, with a fake evaluator (no model training)."""
from __future__ import annotations

import copy

import pytest

from src.agent.policy import Decide, Done, Propose, next_action
from src.agent.store import TrialStore, fingerprint
from src.utils.config import load_config

CFG = load_config("conf/agent.yaml")
HEADER = {"seed": 0, "mode": "dry_run", "models": ["a"], "code": "x", "features": {"a": "1"}, "config": "c"}


def fake_auroc(recipe: str, phase: int, size: int | None, table: dict) -> float:
    base = table.get(recipe, 0.70)
    return base + (0.01 if phase == 2 else 0) + {None: 0, 32: 0, 64: table.get("bigger", -0.005), 16: 0.002}[size]


def fake_compare(records: list[dict]):
    by_id = {r["id"]: r for r in records if r["type"] == "experiment"}

    def compare(a: str, b: str) -> dict:
        d = by_id[a]["scores"]["macro_auroc"]["oof"] - by_id[b]["scores"]["macro_auroc"]["oof"]
        return {"gain": d, "ci": [d - 0.05, d + 0.05], "share_won": 0.99 if d > 0.03 else 0.6}
    return compare


def run(store: TrialStore, table: dict, cfg: dict = CFG, max_steps: int | None = None) -> Done | None:
    """The stage-10 loop with a fake evaluator; stops after `max_steps` actions if given."""
    store.open_run(HEADER)
    steps = 0
    while not isinstance(a := next_action(store.records(), cfg, fake_compare(store.records())), Done):
        if max_steps is not None and steps == max_steps:
            return None
        if isinstance(a, Decide):
            store.append({"type": "decision", "id": store.next_id("D"), "rule": a.rule, "based_on": a.based_on,
                          "statistics": a.statistics, "action": a.action, "reason": a.reason})
        else:
            assert isinstance(a, Propose)
            v = fake_auroc(a.recipe, a.phase, a.size, table)
            store.append({"type": "experiment", "id": store.next_id("T"), "phase": a.phase, "recipe": a.recipe,
                          "size": a.size, "status": "done", "based_on": a.based_on, "reason": a.reason,
                          "fingerprint": fingerprint({"r": a.recipe, "p": a.phase, "s": a.size}),
                          "scores": {"macro_auroc": {"oof": v}, "balanced_accuracy": {"oof": 0.4}}})
        steps += 1
    return a


def test_full_search_follows_the_protocol(tmp_path):
    store = TrialStore(tmp_path / "t.jsonl", 0)
    done = run(store, {"L2": 0.80, "N2": 0.79, "E1": 0.78})
    recs = store.records()
    exps = [r for r in recs if r["type"] == "experiment"]
    p1 = [e["recipe"] for e in exps if e["phase"] == 1]
    assert p1 == CFG["order"] and len(exps) <= CFG["search"]["budget"]
    short = next(r for r in recs if r["type"] == "decision" and r["rule"] == "shortlist")
    assert short["action"]["shortlist"] == ["L2", "N2", "E1"]
    p2 = [(e["recipe"], e["size"]) for e in exps if e["phase"] == 2]
    assert p2 == [("L2", None), ("N2", 32), ("N2", 64), ("N2", 16), ("E1", None)]
    assert isinstance(done, Done) and recs[-1]["rule"] == "final_selection"
    seen = set()
    for r in recs[1:]:                                  # every citation points to an earlier record
        assert set(r["based_on"]) <= seen
        seen.add(r["id"])
    assert len({e["fingerprint"] for e in exps}) == len(exps)


def test_size_path_stops_when_the_larger_size_helps(tmp_path):
    store = TrialStore(tmp_path / "t.jsonl", 0)
    run(store, {"N1": 0.85, "bigger": +0.01})
    sizes = [e["size"] for e in store.records() if e["type"] == "experiment" and e["phase"] == 2 and e["recipe"] == "N1"]
    assert sizes == [32, 64]


def test_simpler_recipe_wins_unless_clearly_beaten(tmp_path):
    # L2 is best but within 0.03 of E1 (not clearly better) -> the simpler E1 is selected.
    store = TrialStore(tmp_path / "t.jsonl", 0)
    run(store, {"L2": 0.80, "E1": 0.79})
    final = store.records()[-1]
    assert final["action"]["candidate"] != final["action"]["winner"]
    assert final["action"]["winner_recipe"] == "E1"
    # Far ahead -> the candidate wins.
    store2 = TrialStore(tmp_path / "t2.jsonl", 0)
    run(store2, {"L2": 0.90, "E1": 0.79})
    assert store2.records()[-1]["action"]["winner_recipe"] == "L2"


def test_budget_is_never_exceeded(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["search"]["budget"] = 13
    store = TrialStore(tmp_path / "t.jsonl", 0)
    run(store, {"N1": 0.80, "N2": 0.80, "N3": 0.80}, cfg)
    recs = store.records()
    assert sum(r["type"] == "experiment" for r in recs) == 13
    assert "Budget of 13" in recs[-1]["reason"]


def test_resume_gives_the_same_record(tmp_path):
    table = {"L2": 0.80, "N2": 0.79, "E1": 0.78}
    a = TrialStore(tmp_path / "a.jsonl", 0)
    run(a, table)
    b = TrialStore(tmp_path / "b.jsonl", 0)
    assert run(b, table, max_steps=7) is None          # interrupted
    run(b, table)                                       # restarted
    assert a.records() == b.records()


def test_resume_refuses_a_different_version(tmp_path):
    store = TrialStore(tmp_path / "t.jsonl", 0)
    store.open_run(HEADER)
    with pytest.raises(SystemExit):
        store.open_run(HEADER | {"code": "other"})
