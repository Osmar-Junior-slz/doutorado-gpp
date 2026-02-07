import json

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline


def _run_dummy_pipeline(tmp_path):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    out_dir = tmp_path / "out"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))
    return out_dir


def test_summary_json_is_written_and_has_required_keys(tmp_path):
    out_dir = _run_dummy_pipeline(tmp_path)
    summary_path = out_dir / "summary.json"

    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    required_keys = {
        "run_id",
        "mode",
        "n_pockets_detected",
        "n_pockets_used",
        "best_score_cheap",
        "best_score_expensive",
        "expensive_ran_count",
        "expensive_skipped_count",
        "n_eval_total",
        "best_cheap_by_pocket",
        "best_expensive_by_pocket",
        "config_resolved_subset",
    }

    assert required_keys.issubset(payload.keys())


def test_report_json_compare_schema(tmp_path):
    full_summary = tmp_path / "full_summary.json"
    reduced_summary = tmp_path / "reduced_summary.json"
    full_summary.write_text("{}", encoding="utf-8")
    reduced_summary.write_text("{}", encoding="utf-8")

    report = {
        "mode": "compare",
        "full": {
            "summary_path": str(full_summary),
        },
        "reduced": {
            "summary_path": str(reduced_summary),
        },
        "diff": {},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload.get("mode") == "compare"
    assert "summary_path" in payload.get("full", {})
    assert "summary_path" in payload.get("reduced", {})
    assert "diff" in payload


def test_metrics_jsonl_has_required_fields(tmp_path):
    out_dir = _run_dummy_pipeline(tmp_path)
    metrics_path = out_dir / "metrics.jsonl"

    assert metrics_path.exists()

    pocket_metric_names = {"best_score", "mean_score", "n_eval", "n_clashes"}
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            assert "step" in record
            assert "name" in record
            assert "value" in record

            pocket_index = record.get("pocket_index")
            generation = record.get("generation")

            if pocket_index is not None:
                assert isinstance(pocket_index, int)
            if generation is not None:
                assert isinstance(generation, int)
            if record.get("name") in pocket_metric_names:
                assert pocket_index is not None
                assert generation is not None
