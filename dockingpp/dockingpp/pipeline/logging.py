"""Logging estruturado de execução em JSONL/JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dockingpp.reporting.models import ArtifactsManifest, MetricEvent, RunSummary


@dataclass
class RunLogger:
    run_id: str
    out_dir: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(self.out_dir) / "metrics.jsonl"

    def emit_event(self, event: str, **kwargs: Any) -> None:
        payload = MetricEvent(event=event, run_id=self.run_id, **kwargs).model_dump(mode="json")
        self.records.append(payload)
        with self.metrics_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(payload) + "\n")

    def log_metric(self, name: str, value: float, step: int, extra: dict[str, Any] | None = None) -> None:
        context = dict(extra or {})
        context["metric_name"] = name
        context["metric_value"] = value
        event = "search_iteration"
        if name in {"expensive_ran", "expensive_skipped"}:
            event = "expensive_eval" if name == "expensive_ran" else "trigger_expensive"
        self.emit_event(
            event,
            step=step,
            value=value,
            pocket_index=context.get("pocket_index"),
            pocket_id=context.get("pocket_id"),
            reason=context.get("reason"),
            context=context,
        )

    def save_summary(self, summary: RunSummary) -> None:
        path = Path(self.out_dir) / "summary.json"
        path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")

    def save_manifest(self, status: str) -> None:
        manifest = ArtifactsManifest(
            run_id=self.run_id,
            out_dir=self.out_dir,
            status=status,
            files={
                "summary": "summary.json",
                "metrics": "metrics.jsonl",
                "manifest": "artifacts_manifest.json",
            },
        )
        path = Path(self.out_dir) / "artifacts_manifest.json"
        path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")

    def safe_log_error(self, exc: Exception) -> None:
        self.emit_event(
            "run_failed",
            reason=type(exc).__name__,
            context={"error": str(exc)},
            ts=datetime.utcnow(),
        )
