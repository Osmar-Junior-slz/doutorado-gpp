"""Helpers to load run reports and metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

JsonKind = Literal["compare", "single", "unknown"]


@dataclass(frozen=True)
class ReportBundle:
    """Bundle of report data loaded from a folder or upload."""

    kind: JsonKind
    main_json: dict[str, Any]
    metrics: list[dict[str, Any]] | None
    aux_jsons: dict[str, dict[str, Any]]


def find_runs(root_dir: Path) -> list[Path]:
    """Find run folders containing at least one JSON or JSONL file."""

    if not root_dir.exists():
        return []
    runs = [
        child
        for child in root_dir.iterdir()
        if child.is_dir()
        and (any(child.glob("*.json")) or any(child.glob("*.jsonl")))
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


def infer_json_kind(obj: dict[str, Any]) -> JsonKind:
    """Infer the report kind based on known keys."""

    if "full" in obj and "reduced" in obj:
        return "compare"
    runs = obj.get("runs")
    if isinstance(runs, dict) and "full" in runs and "reduced" in runs:
        return "compare"
    comparison = obj.get("comparison")
    if isinstance(comparison, dict) and "full" in comparison and "reduced" in comparison:
        return "compare"
    if obj.get("mode") == "compare":
        return "compare"

    if any(key in obj for key in ("best_score_cheap", "best_score")):
        return "single"
    if any(key in obj for key in ("n_eval", "evaluations")):
        return "single"
    if any(key in obj for key in ("pose", "best_pose")):
        return "single"
    if any(key in obj for key in ("config", "cfg")):
        return "single"

    return "unknown"


def summarize_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize metrics records by name/value."""

    summary: dict[str, Any] = {}
    for record in records:
        name = record.get("name")
        if not name:
            continue
        summary[name] = record.get("value")
    return summary


def metrics_series(
    records: list[dict[str, Any]],
    keys: list[str],
    step_keys: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build a series for charting from metrics records."""

    if not records:
        return [], None
    if step_keys is None:
        step_keys = ["generation", "gen", "step", "iter"]

    selected_key = next((key for key in keys if any(rec.get("name") == key for rec in records)), None)
    series: list[dict[str, Any]] = []
    if selected_key:
        for idx, record in enumerate(records):
            if record.get("name") != selected_key:
                continue
            step = next((record.get(key) for key in step_keys if record.get(key) is not None), None)
            if step is None:
                step = idx
            series.append({"step": step, "score": record.get("value")})
        return series, selected_key

    selected_key = next((key for key in keys if any(key in rec for rec in records)), None)
    if not selected_key:
        return [], None

    for idx, record in enumerate(records):
        if selected_key not in record:
            continue
        step = next((record.get(key) for key in step_keys if record.get(key) is not None), None)
        if step is None:
            step = idx
        series.append({"step": step, "score": record.get(selected_key)})
    return series, selected_key


def _extract_compare_block(report_data: dict[str, Any], label: str) -> dict[str, Any] | None:
    for container in (report_data, report_data.get("runs"), report_data.get("comparison")):
        if isinstance(container, dict) and label in container and isinstance(container[label], dict):
            return container[label]
    return None


def _lookup_metric(block: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in block and block[key] is not None:
            return block[key]
    metrics = block.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]
    return None


def build_compare_table(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build table rows for full vs reduced comparisons."""

    rows: list[dict[str, Any]] = []
    for label in ("full", "reduced"):
        block = _extract_compare_block(report_data, label)
        if not block:
            continue
        label_name = "Completo" if label == "full" else "Reduzido"
        rows.append(
            {
                "Modo": label_name,
                "Melhor score (cheap)": _lookup_metric(block, ["best_score_cheap", "best_score", "best"]),
                "Avaliações": _lookup_metric(block, ["n_eval", "evals", "evaluations"]),
                "Bolsões totais": _lookup_metric(block, ["n_pockets_total", "pockets_total"]),
                "Bolsões usados": _lookup_metric(block, ["n_pockets_used", "pockets_used"]),
                "Razão de redução": _lookup_metric(block, ["reduction_ratio", "ratio"]),
                "Tempo (s)": _lookup_metric(block, ["elapsed_s", "elapsed_seconds", "elapsed"]),
            }
        )
    return rows
