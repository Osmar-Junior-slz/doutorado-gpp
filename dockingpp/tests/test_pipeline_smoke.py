import json

from dockingpp.data.io import load_config, load_peptide, load_pockets, load_receptor
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


def test_pipeline_pocket_reduction(tmp_path):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    cfg.full_search = False
    cfg.top_pockets = 1
    out_dir = tmp_path / "out_reduced"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    metrics = []
    with open(out_dir / "metrics.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            metrics.append(json.loads(line))

    metric_map = {entry["name"]: entry["value"] for entry in metrics}
    total_pockets = int(metric_map.get("n_pockets_total", 0))
    used_pockets = int(metric_map.get("n_pockets_used", 0))
    reduction_ratio = float(metric_map.get("reduction_ratio", 0.0))

    assert used_pockets < total_pockets
    assert reduction_ratio > 0


def test_load_pockets_dummy_global():
    receptor = load_receptor("__dummy__")
    peptide = load_peptide("__dummy__")

    pockets = load_pockets(receptor)

    assert peptide.get("dummy") is True
    assert len(pockets) >= 1
    assert pockets[0].id == "global"
