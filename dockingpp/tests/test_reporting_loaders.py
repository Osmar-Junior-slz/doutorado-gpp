import json
from pathlib import Path

from dockingpp.reporting.loaders import load_report_bundle


def test_loader_summary_metrics_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run1"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "a",
                "status": "success",
                "mode": "single",
                "engine": "e",
                "receptor": "r",
                "peptide": "p",
                "runtime_sec": 1,
                "omega_full": 2,
                "omega_reduced": 1,
                "omega_ratio": 0.5,
                "n_pockets_total": 2,
                "n_pockets_selected": 1,
                "n_evals_cheap": 5,
                "n_evals_expensive": 1,
                "best_score_cheap": 0.2,
                "best_score_expensive": 0.1,
                "confidence_final": 0.9,
                "trigger_count_expensive": 1,
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(json.dumps({"event": "run_started", "run_id": "a"}) + "\n", encoding="utf-8")
    bundle = load_report_bundle(run)
    assert bundle.summary.run_id == "a"


def test_loader_legacy_format(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run_id": "old", "n_pockets_detected": 8, "n_eval_total": 100}), encoding="utf-8")
    bundle = load_report_bundle(run)
    assert bundle.summary.n_pockets_total == 8
