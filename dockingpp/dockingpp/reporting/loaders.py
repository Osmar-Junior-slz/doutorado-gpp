"""Loaders para artefatos de execução de relatórios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dockingpp.reporting.compat import normalize_legacy_metric, normalize_legacy_summary
from dockingpp.reporting.models import ReportBundle
from dockingpp.reporting.schema import build_bundle


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs = {p.parent for p in root.rglob("summary.json")}
    if (root / "summary.json").exists():
        dirs.add(root)
    return sorted(dirs)


def load_report_bundle(run_dir: Path) -> ReportBundle:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    manifest_path = run_dir / "artifacts_manifest.json"

    raw_summary = load_json(summary_path) if summary_path.exists() else {}
    summary = normalize_legacy_summary(raw_summary, run_id=raw_summary.get("run_id"))
    metrics_raw = load_jsonl(metrics_path)
    metrics = [normalize_legacy_metric(item, run_id=summary["run_id"]) for item in metrics_raw]
    manifest = load_json(manifest_path) if manifest_path.exists() else None
    return build_bundle(summary=summary, metrics=metrics, manifest=manifest)
