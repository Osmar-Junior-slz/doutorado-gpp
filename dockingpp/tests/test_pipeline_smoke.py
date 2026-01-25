import json

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.data.structs import RunResult


def test_pipeline_smoke(tmp_path):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    out_dir = tmp_path / "out"
    out_dir_repeat = tmp_path / "out_repeat"

    result = run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))
    result_repeat = run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir_repeat))

    assert isinstance(result, RunResult)
    assert (out_dir / "result.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir_repeat / "metrics.jsonl").exists()

    with open(out_dir / "result.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload["best_score_cheap"], float)
    assert payload["best_score_cheap"] == result.best_pose.score_cheap

    with open(out_dir_repeat / "result.json", "r", encoding="utf-8") as handle:
        repeat_payload = json.load(handle)
    assert repeat_payload["best_score_cheap"] == payload["best_score_cheap"]
    assert result_repeat.best_pose.score_cheap == result.best_pose.score_cheap
