import json

import pytest

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, _normalize_search_space_mode, run_pipeline
from dockingpp.reporting.loaders import find_report_runs


def _run_reduced(tmp_path, **overrides):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.scan["enabled"] = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    out_dir = tmp_path / "reduced"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))
    return out_dir


def test_reduced_creates_one_run_per_selected_pocket(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    for pocket_id in summary["selected_pockets"]:
        assert (out_dir / pocket_id / "result.json").exists()
        assert (out_dir / pocket_id / "summary.json").exists()


def test_reduced_parent_summary_does_not_sum_scores(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    pocket_scores = [item["best_score_cheap"] for item in summary["per_pocket_results"]]
    assert summary["best_over_pockets_cheap"] == max(pocket_scores)


def test_compare_uses_best_pocket_not_sum(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["compare_policy"] == "best_pocket_vs_full"
    assert summary["best_pocket_id"] in summary["selected_pockets"]


def test_split_budget_divides_evals_across_pockets(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["budget_policy"] == "split"
    evals = [item["n_eval_total"] for item in summary["per_pocket_results"]]
    assert sum(evals) == summary["total_n_eval"]


def test_infeasible_pockets_are_not_selected(tmp_path):
    out_dir = _run_reduced(tmp_path, top_pockets=3)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rejected = summary.get("rejected_pockets", [])
    assert isinstance(rejected, list)


def test_fallback_to_full_when_no_feasible_pocket(tmp_path):
    out_dir = _run_reduced(tmp_path, top_pockets=3)
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.scan["enabled"] = True
    cfg.scan["max_clash_ratio"] = -1.0
    out_dir = tmp_path / "reduced_fallback"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["fallback_to_full"] is True
    assert (out_dir / "fallback_full" / "result.json").exists()


def test_schema_uses_only_full_and_reduced(tmp_path):
    assert _normalize_search_space_mode("global", True) == "full"
    assert _normalize_search_space_mode("pockets", False) == "reduced"


def test_per_pocket_artifacts_are_written(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    for pocket_id in summary["selected_pockets"]:
        assert (out_dir / pocket_id / "metrics.jsonl").exists()


def test_reduced_aggregate_summary_contains_best_pocket(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "reduced_aggregate"
    assert summary["best_pocket_id"] is not None


def test_report_and_gui_read_new_reduced_schema(tmp_path):
    out_dir = _run_reduced(tmp_path)
    runs = find_report_runs(tmp_path)
    assert runs
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "per_pocket_results" in summary


def test_reduced_aggregate_does_not_write_legacy_flat_best_score_fields(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "best_score_cheap" not in summary
    assert "best_score_expensive" not in summary
    assert "best_over_pockets_cheap" in summary
    assert "best_over_pockets_expensive" in summary


def test_reduced_aggregate_budget_accounting_fields(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "total_eval_budget_requested" in summary
    assert "total_eval_budget_assigned" in summary
    assert "budget_delta" in summary
    assert summary["budget_delta"] == summary["total_eval_budget_assigned"] - summary["total_eval_budget_requested"]


def test_reduced_aggregate_has_schema_version(tmp_path):
    out_dir = _run_reduced(tmp_path)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary.get("schema_version") == "2.0"


def test_reduced_aggregate_runtime_breakdown_counts_global_once(tmp_path):
    out_dir = _run_reduced(tmp_path, top_pockets=2, scan={"enabled": False})
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))

    assert len(summary["per_pocket_results"]) >= 2
    assert "pocketing_time_sec" in summary
    assert "scan_time_sec" in summary
    assert "search_time_total_sec" in summary

    assert all("search_time_sec" in item for item in summary["per_pocket_results"])
    search_sum = sum(item["search_time_sec"] for item in summary["per_pocket_results"])
    assert summary["search_time_total_sec"] == pytest.approx(search_sum)
    assert summary["total_runtime_sec"] == pytest.approx(
        summary["pocketing_time_sec"] + summary["scan_time_sec"] + summary["search_time_total_sec"]
    )

    for key in ("pocketing_time_sec", "scan_time_sec", "search_time_total_sec", "total_runtime_sec"):
        assert result[key] == pytest.approx(summary[key])

    metric_records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics_by_name = {item.get("name"): item.get("value") for item in metric_records}
    assert metrics_by_name["runtime.pocketing_time_sec"] == pytest.approx(summary["pocketing_time_sec"])
    assert metrics_by_name["runtime.scan_time_sec"] == pytest.approx(summary["scan_time_sec"])
    assert metrics_by_name["runtime.search_time_total_sec"] == pytest.approx(summary["search_time_total_sec"])
    assert metrics_by_name["runtime.total_runtime_sec"] == pytest.approx(summary["total_runtime_sec"])
