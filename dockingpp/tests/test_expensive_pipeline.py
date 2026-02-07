import json

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline


def test_pipeline_does_not_crash_with_expensive_enabled(tmp_path):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    cfg.expensive_every = 1
    cfg.expensive_topk = 1
    out_dir = tmp_path / "out_expensive_enabled"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    metrics = []
    with open(out_dir / "metrics.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            metrics.append(json.loads(line))

    metric_names = {entry["name"] for entry in metrics}
    assert {"expensive_ran", "expensive_skipped"} & metric_names
