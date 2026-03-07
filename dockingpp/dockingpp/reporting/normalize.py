"""Normalização de dados para consumo da UI."""

from __future__ import annotations

from typing import Any

from dockingpp.reporting.models import ReportBundle


def optional_warnings(bundle: ReportBundle) -> list[str]:
    warnings = list(bundle.warnings)
    if bundle.summary.best_score_expensive is None:
        warnings.append("score caro não executado")
    if bundle.summary.confidence_final is None:
        warnings.append("calibração ausente")
    if bundle.summary.legacy.get("rmsd") is None:
        warnings.append("RMSD indisponível")
    return warnings


def to_timeseries(bundle: ReportBundle) -> dict[str, list[Any]]:
    steps: list[int] = []
    best_cheap: list[float | None] = []
    expensive_values: list[float | None] = []
    trigger_steps: list[int] = []

    for ev in bundle.metrics:
        if ev.step is not None:
            steps.append(int(ev.step))
        if ev.event == "search_iteration":
            best_cheap.append(ev.context.get("best_score_cheap", ev.value))
        if ev.event == "expensive_eval":
            expensive_values.append(ev.value)
        if ev.event == "trigger_expensive" and ev.step is not None:
            trigger_steps.append(int(ev.step))

    return {
        "iter": steps,
        "best_cheap": best_cheap,
        "best_expensive": expensive_values,
        "trigger_timeline": trigger_steps,
    }
