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
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    required = {
        "run_id",
        "status",
        "mode",
        "engine",
        "receptor",
        "peptide",
        "runtime_sec",
        "omega_full",
        "omega_reduced",
        "omega_ratio",
        "n_pockets_total",
        "n_pockets_selected",
        "n_evals_cheap",
        "n_evals_expensive",
        "best_score_cheap",
        "best_score_expensive",
        "confidence_final",
        "trigger_count_expensive",
    }
    assert required.issubset(payload.keys())


def test_metrics_and_manifest_are_written(tmp_path):
    out_dir = _run_dummy_pipeline(tmp_path)
    metrics = out_dir / "metrics.jsonl"
    manifest = out_dir / "artifacts_manifest.json"
    assert metrics.exists()
    assert manifest.exists()

    first = json.loads(metrics.read_text(encoding="utf-8").splitlines()[0])
    assert "event" in first
    assert "run_id" in first
