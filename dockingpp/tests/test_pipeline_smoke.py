from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.data.structs import RunResult


def test_pipeline_smoke(tmp_path):
    cfg_data = load_config("configs/default.yaml")
    cfg = Config(**cfg_data)
    out_dir = tmp_path / "out"

    result = run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    assert isinstance(result, RunResult)
    assert (out_dir / "result.json").exists()
