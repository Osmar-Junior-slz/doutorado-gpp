from __future__ import annotations

import json
from pathlib import Path

from dockingpp.pipeline.run import Config, run_pipeline


def _base_cfg() -> Config:
    cfg = Config(seed=1, generations=2, pop_size=3, topk=1)
    cfg.debug_enabled = True
    cfg.debug_level = "AUDIT"
    cfg.search_space_mode = "full"
    cfg.full_search = True
    return cfg


def _run(tmp_path: Path, *, reduced: bool = False, debug_enabled: bool = True, debug_level: str = "AUDIT", fallback: bool = False):
    cfg = _base_cfg()
    cfg.debug_enabled = debug_enabled
    cfg.debug_level = debug_level
    if reduced:
        cfg.search_space_mode = "reduced"
        cfg.full_search = False
        cfg.top_pockets = 3
        cfg.scan = {"enabled": True, "max_clash_ratio": -1.0 if fallback else 1.0, "select_top_k": 2}
    out = tmp_path / ("reduced" if reduced else "full")
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))
    return out


def test_debug_folder_is_created_when_enabled(tmp_path):
    out = _run(tmp_path, debug_enabled=True)
    assert (out / "debug").exists()


def test_debug_folder_is_not_created_when_disabled(tmp_path):
    out = _run(tmp_path, debug_enabled=False)
    assert not (out / "debug").exists()


def test_debug_manifest_is_written(tmp_path):
    out = _run(tmp_path)
    assert (out / "debug" / "manifest.json").exists()


def test_debug_summary_is_written_inside_debug_folder(tmp_path):
    out = _run(tmp_path)
    assert (out / "debug" / "debug_summary.json").exists()


def test_trace_files_are_written_inside_debug_folder(tmp_path):
    out = _run(tmp_path)
    for name in ["trace.jsonl", "decision_trace.jsonl", "warnings.jsonl", "errors.jsonl"]:
        assert (out / "debug" / name).exists()


def test_reduced_writes_per_pocket_debug_files(tmp_path):
    out = _run(tmp_path, reduced=True)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    for pocket_id in summary.get("selected_pockets", []):
        assert (out / "debug" / "pockets" / pocket_id / "trace.jsonl").exists()
        assert (out / "debug" / "pockets" / pocket_id / "decision_trace.jsonl").exists()
        assert (out / "debug" / "pockets" / pocket_id / "debug_summary.json").exists()


def test_fallback_is_recorded_in_debug_summary(tmp_path):
    out = _run(tmp_path, reduced=True, fallback=True)
    payload = json.loads((out / "debug" / "debug_summary.json").read_text(encoding="utf-8"))
    assert payload["fallback_to_full"] is True


def test_manifest_lists_generated_debug_files(tmp_path):
    out = _run(tmp_path)
    payload = json.loads((out / "debug" / "manifest.json").read_text(encoding="utf-8"))
    assert "debug/trace.jsonl" in payload["files"]


def test_trace_events_have_required_fields(tmp_path):
    out = _run(tmp_path)
    lines = (out / "debug" / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    evt = json.loads(lines[0])
    required = {
        "schema_version",
        "run_id",
        "event_id",
        "parent_event_id",
        "stage",
        "substage",
        "event_type",
        "timestamp",
        "monotonic_ms",
        "level",
        "module",
        "function",
        "search_space_mode",
        "pocket_id",
        "engine",
        "payload",
    }
    assert required.issubset(evt.keys())


def test_pocket_rejection_reason_is_traced(tmp_path):
    out = _run(tmp_path, reduced=True, fallback=True)
    lines = (out / "debug" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line).get("event_type") == "pocket_rejected" for line in lines)


def test_budget_split_is_traced(tmp_path):
    out = _run(tmp_path, reduced=True)
    lines = (out / "debug" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line).get("event_type") == "budget_split" for line in lines)


def test_run_finished_contains_final_status(tmp_path):
    out = _run(tmp_path)
    lines = (out / "debug" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    matches = [json.loads(line) for line in lines if json.loads(line).get("event_type") == "run_finished"]
    assert matches


def test_debug_level_off_disables_trace_output(tmp_path):
    out = _run(tmp_path, debug_enabled=True, debug_level="OFF")
    assert not (out / "debug").exists()


def test_debug_level_audit_enables_detailed_events(tmp_path):
    out = _run(tmp_path, debug_enabled=True, debug_level="AUDIT")
    lines = (out / "debug" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line).get("event_type") == "generation_completed" for line in lines)
