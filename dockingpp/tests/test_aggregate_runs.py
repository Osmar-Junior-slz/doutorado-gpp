import csv
import json

from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.reporting.aggregate_runs import _to_row, aggregate_runs


def test_aggregate_runs_generates_comparison_csv(tmp_path):
    out_a = tmp_path / "runs" / "a"
    out_b = tmp_path / "runs" / "b"

    run_pipeline(Config(seed=7, full_search=True), "__dummy__", "__dummy__", str(out_a))
    run_pipeline(Config(seed=13, full_search=False), "__dummy__", "__dummy__", str(out_b))

    csv_path = aggregate_runs(tmp_path / "runs")
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "run_id",
        "complex_id",
        "seed",
        "search_space_mode",
        "runtime_sec",
        "n_eval_total",
        "n_pockets_total",
        "n_pockets_used",
        "reduction_ratio",
        "best_score_cheap",
        "best_score_expensive",
    }


def test_summary_has_traceability_top_level_fields(tmp_path):
    out_dir = tmp_path / "out"
    run_pipeline(Config(seed=11, expensive_every=2, expensive_topk=3), "__dummy__", "__dummy__", str(out_dir))

    with open(out_dir / "summary.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected = {
        "run_id",
        "input_id",
        "seed",
        "search_space_mode",
        "runtime_sec",
        "search_time_sec",
        "pocketing_sec",
        "scan_sec",
        "n_eval_total",
        "n_pockets_total",
        "n_pockets_used",
        "reduction_ratio",
        "best_score_cheap",
        "best_score_expensive",
        "expensive_enabled",
        "expensive_policy",
    }
    assert expected.issubset(payload.keys())
    assert payload["seed"] == 11
    assert payload["expensive_enabled"] is True
    assert payload["expensive_policy"]["every"] == 2
    assert payload["expensive_policy"]["topk"] == 3


def test_aggregate_runs_maps_reduced_aggregate_fields():
    row = _to_row(
        {
            "run_id": "run-reduced",
            "search_space_mode": "reduced",
            "total_runtime_sec": 3.5,
            "total_n_eval": 42,
            "n_pockets_total": 10,
            "n_pockets_used": 2,
            "best_over_pockets_cheap": 1.2,
            "best_over_pockets_expensive": 0.9,
        }
    )

    assert row["runtime_sec"] == 3.5
    assert row["n_eval_total"] == 42
    assert row["best_score_cheap"] == 1.2
    assert row["best_score_expensive"] == 0.9
