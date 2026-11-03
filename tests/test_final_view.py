"""src/eval/final_view.py: tables and figures read the committed final results correctly."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from src.eval import final_view as fv  # noqa: E402

needs_results = pytest.mark.skipif(not Path("artifacts/final/seed_0/results.json").is_file(),
                                   reason="final results not present")


@needs_results
def test_tables_have_the_task_rows_and_all_seeds():
    runs = fv.load_final("artifacts/final")
    assert sorted(runs) == [0, 1, 2, 3, 4] and fv.primary(runs) == "H1"
    t = fv.task_table(runs, "H1")
    assert list(t.index) == [fv.ROW_NAMES[r] for r in fv.TABLE_ROWS] and "Seed 0 AUROC [95% CI]" in t
    assert t.loc["Fused + age and sex", "Test macro AUROC (mean ± sd)"].startswith("0.894")
    g = fv.gain_table(runs, "H1")
    assert g.index.is_unique and len(g) == 11
    assert list(fv.designs_table(runs).index) == ["H1", "H2", "C1"]


@needs_results
def test_identical_runs_compare_to_zero_and_figures_render():
    runs = fv.load_final("artifacts/final")
    assert (fv.compare_runs(runs, runs)["Largest difference"] == 0).all()
    for f in (fv.plot_per_seed, fv.plot_gains, fv.plot_per_class, fv.plot_confusion):
        assert f(runs, "H1").axes
    v = fv.versions_table("artifacts/versions")
    assert v.loc["v11", "Agent's winner"] == "H1:raw" and v.loc["v11", "Current best"] == "v11"
    assert fv.plot_versions(v).axes and len(fv.decisions("v11")) == 6
