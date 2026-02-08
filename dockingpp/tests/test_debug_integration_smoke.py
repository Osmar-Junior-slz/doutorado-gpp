"""Teste smoke para integração do debug logger no pipeline."""

from __future__ import annotations

from typing import Any

from dockingpp.pipeline import run as run_module


def test_debug_integration_smoke(tmp_path, monkeypatch):
    """PT-BR: valida emissão de eventos principais no pipeline."""

    events: list[dict[str, Any]] = []

    class DummyDebugLogger:
        def __init__(self, enabled: bool, path: str, level: str = "INFO") -> None:
            self.enabled = enabled
            self.path = path
            self.level = level
            self.run_id = None

        def log(self, event: dict[str, Any], level: str = "INFO") -> None:
            events.append(event)

        def close(self) -> None:
            return None

    monkeypatch.setattr(run_module, "DebugLogger", DummyDebugLogger)

    cfg = run_module.Config(
        seed=1,
        generations=1,
        pop_size=2,
        topk=1,
        debug_log_enabled=True,
        debug_log_path=str(tmp_path / "debug.jsonl"),
    )
    run_module.run_pipeline(cfg, "__dummy__", "__dummy__", str(tmp_path))

    event_types = {event.get("type") for event in events}
    assert "config_resolved" in event_types
    assert "pocket_selection" in event_types
