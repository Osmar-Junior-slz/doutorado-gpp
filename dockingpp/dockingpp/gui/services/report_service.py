"""Helpers to load run reports and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def find_runs(root_dir: Path) -> list[Path]:
    """Find run folders containing result.json or report.json."""

    if not root_dir.exists():
        return []
    runs = [
        child
        for child in root_dir.iterdir()
        if child.is_dir() and ((child / "result.json").exists() or (child / "report.json").exists())
    ]
    return sorted(runs)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def summarize_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize metrics records by name/value."""

    summary: dict[str, Any] = {}
    for record in records:
        name = record.get("name")
        if not name:
            continue
        summary[name] = record.get("value")
    return summary


def metrics_series(records: list[dict[str, Any]], keys: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    """Build a series for charting from metrics records."""

    selected_key = next((key for key in keys if any(rec.get("name") == key for rec in records)), None)
    if not selected_key:
        return [], None

    series: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if record.get("name") != selected_key:
            continue
        step = record.get("step")
        if step is None:
            step = record.get("generation")
        if step is None:
            step = idx
        series.append({"step": step, "score": record.get("value")})
    return series, selected_key


def build_compare_table(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build table rows for full vs reduced comparisons."""

    rows: list[dict[str, Any]] = []
    for label in ("full", "reduced"):
        if label not in report_data:
            continue
        metrics = report_data[label].get("metrics", {})
        label_name = "Completo" if label == "full" else "Reduzido"
        rows.append(
            {
                "Modo": label_name,
                "Melhor score (cheap)": report_data[label].get("best_score_cheap"),
                "Avaliações": metrics.get("n_eval"),
                "Bolsões totais": metrics.get("n_pockets_total"),
                "Bolsões usados": metrics.get("n_pockets_used"),
                "Razão de redução": metrics.get("reduction_ratio"),
                "Tempo (s)": report_data[label].get("elapsed_seconds"),
            }
        )
    return rows

