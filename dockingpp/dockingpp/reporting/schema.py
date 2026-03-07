"""Validação de schema dos artefatos de relatório."""

from __future__ import annotations

from typing import Any

from dockingpp.reporting.models import ArtifactsManifest, MetricEvent, ReportBundle, RunSummary


REQUIRED_EVENTS = {
    "run_started",
    "pocket_ranked",
    "pocket_selected",
    "search_iteration",
    "candidate_promoted",
    "trigger_expensive",
    "expensive_eval",
    "confidence_updated",
    "run_finished",
    "run_failed",
}


def validate_summary(payload: dict[str, Any]) -> RunSummary:
    return RunSummary.model_validate(payload)


def validate_metric_event(payload: dict[str, Any]) -> MetricEvent:
    return MetricEvent.model_validate(payload)


def validate_manifest(payload: dict[str, Any]) -> ArtifactsManifest:
    return ArtifactsManifest.model_validate(payload)


def build_bundle(summary: dict[str, Any], metrics: list[dict[str, Any]], manifest: dict[str, Any] | None) -> ReportBundle:
    summary_model = validate_summary(summary)
    metric_models = [validate_metric_event(item) for item in metrics]
    manifest_model = validate_manifest(manifest) if manifest else None
    warnings: list[str] = []
    seen = {m.event for m in metric_models}
    missing = sorted(REQUIRED_EVENTS - seen)
    if missing:
        warnings.append(f"Eventos ausentes: {', '.join(missing)}")
    return ReportBundle(summary=summary_model, metrics=metric_models, manifest=manifest_model, warnings=warnings)
