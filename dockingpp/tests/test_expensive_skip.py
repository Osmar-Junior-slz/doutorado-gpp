import json

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.scoring import expensive as expensive_mod


def test_expensive_skipped_logged(tmp_path):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    cfg.expensive_every = 1
    cfg.expensive_topk = 1
    out_dir = tmp_path / "out_expensive_skip"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    metrics = []
    with open(out_dir / "metrics.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            metrics.append(json.loads(line))

    metric_names = {entry["name"] for entry in metrics}
    assert "expensive_skipped" in metric_names


def test_expensive_exception_is_logged_and_does_not_crash(tmp_path, monkeypatch):
    def _raise_value_error(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(expensive_mod, "_score_pose_expensive_impl", _raise_value_error)
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    cfg.expensive_every = 1
    cfg.expensive_topk = None
    out_dir = tmp_path / "out_expensive_exception"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    metrics = []
    with open(out_dir / "metrics.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            metrics.append(json.loads(line))

    assert any(
        entry["name"] == "expensive_skipped"
        and entry.get("reason") == "exception"
        and entry.get("error") == "ValueError"
        for entry in metrics
    )
