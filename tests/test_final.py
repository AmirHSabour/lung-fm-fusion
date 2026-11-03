"""Stage 12 (final evaluation): split integrity, no use of test-set statistics, frozen configuration."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.splits import load_split
from src.eval.dataset import load_dev, load_test
from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import build_candidate
from src.fusion.registry import RECIPES
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)

final = importlib.import_module("scripts.12_final_eval")
KW = dict(splits_dir=Path("artifacts/splits"), cohort_csv=Path("artifacts/cohort.csv"),
          features_dir=Path("artifacts/features"))
needs_features = pytest.mark.skipif(not (KW["features_dir"] / "uni2h.npz").is_file(), reason="features not built")


@needs_features
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_dev_and_test_are_disjoint_complete_and_stratified(seed):
    d, t = load_dev(seed, ["uni2h"], **KW), load_test(seed, ["uni2h"], **KW)
    sn = load_split(KW["splits_dir"], seed).set_index("WSI_ID")["SampleNumber"]
    assert (len(d.y), len(t.y)) == (162, 41)
    assert not set(d.wsi_ids) & set(t.wsi_ids) and not set(sn[d.wsi_ids]) & set(sn[t.wsi_ids])
    assert np.all(np.bincount(t.y, minlength=7) > 0) and np.all(t.folds == -1)


def test_block_layout_matches_stage_10():
    assert final.with_id(["uni2h", META]) == ["uni2h", ID, META]
    assert final.with_id(["uni2h", "virchow2"]) == ["uni2h", "virchow2", ID]
    assert final.with_id([META]) == [ID, META]


@pytest.mark.parametrize("cand", ["H1:raw", "H2:raw", "C1:raw"])
def test_test_predictions_use_no_test_set_statistics(cache, cand, tmp_path):  # noqa: F811
    """Fitted on the training rows only: predicting the held-out rows together or one at a time gives the
    same probabilities, so nothing is estimated from the predicted (test) rows."""
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["mexp", "clam", "hier"], "expert_cache": None}
    train = np.arange(len(y)) % 3 != 0
    est = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train])
    together = est.predict_proba(X[~train])
    alone = np.vstack([est.predict_proba(X[~train][i:i + 1]) for i in range(3)])
    assert np.allclose(together[:3], alone, atol=1e-6)


def test_final_config_is_consistent_with_the_loop():
    f = load_config("conf/final.yaml")
    assert [d["role"] for d in f["designs"].values()].count("primary") == 1
    assert f["designs"]["H1"]["role"] == "primary"                      # the agent's own v11 selection
    recs = [json.loads(line) for line in Path("artifacts/versions/v11/seed_0/agent/trials.jsonl").read_text().splitlines()]
    final_sel = next(r for r in recs if r.get("rule") == "final_selection")
    assert final_sel["action"]["winner_candidate"] == f["designs"]["H1"]["candidate"]
    oof = {r["candidate"]: r["scores"]["macro_auroc"]["oof"] for r in recs
           if r.get("type") == "experiment" and r.get("status") == "done" and r.get("mode") == "screen"}
    for d in f["designs"].values():
        assert d["candidate"].split(":")[0] in RECIPES
        assert abs(oof[d["candidate"]] - d["seed0_dev_auroc"]) < 1e-4
    assert f["models"] == load_config(f["search_config"])["models"]
