from __future__ import annotations

import json
from pathlib import Path

from dockingpp.data.io import load_config
from dockingpp.gui.pages.docking import _build_compare_runs, _search_space_mode_from_label
from dockingpp.pipeline.run import Config, run_pipeline


def test_single_mode_uses_canonical_search_space_mode() -> None:
    assert _search_space_mode_from_label("Global") == "full"
    assert _search_space_mode_from_label("Bols\u00f5es") == "reduced"


def test_compare_mode_uses_canonical_search_space_modes(tmp_path: Path) -> None:
    runs = _build_compare_runs(tmp_path, top_pockets=5)
    by_label = {item["label"]: item for item in runs}

    assert by_label["full"]["search_space_mode"] == "full"
    assert by_label["reduced"]["search_space_mode"] == "reduced"
    assert by_label["full"]["full_search"] is True
    assert by_label["reduced"]["full_search"] is False
    assert by_label["reduced"]["top_pockets"] == 5


def test_compare_reduced_run_builds_per_pocket_artifacts(tmp_path: Path) -> None:
    cfg_data = load_config("configs/default.yaml")
    runs = _build_compare_runs(tmp_path, top_pockets=2)
    reduced = next(item for item in runs if item["label"] == "reduced")

    run_cfg = dict(cfg_data)
    run_cfg["full_search"] = reduced["full_search"]
    run_cfg["search_space_mode"] = reduced["search_space_mode"]
    run_cfg["top_pockets"] = reduced["top_pockets"]

    run_pipeline(Config(**run_cfg), "__dummy__", "__dummy__", str(reduced["out_dir"]))

    summary = json.loads((reduced["out_dir"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "reduced_aggregate"
    assert isinstance(summary.get("per_pocket_results"), list)
    assert summary.get("per_pocket_results")
    for pocket_id in summary.get("selected_pockets", []):
        assert (reduced["out_dir"] / pocket_id / "summary.json").exists()
        assert (reduced["out_dir"] / pocket_id / "result.json").exists()
