"""Logging utilities for dockingpp."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunLogger:
    """In-memory logger that writes JSONL on flush."""

    records: List[Dict[str, Any]] = field(default_factory=list)

    def log_metric(self, name: str, value: float, step: int, extra: Optional[Dict[str, Any]] = None) -> None:
        """Store a metric record."""

        payload = {"name": name, "value": value, "step": step}
        if extra:
            payload.update(extra)
        self.records.append(payload)

    def log_global_metrics(self, total_pockets: int, used_pockets: int) -> None:
        """Store global metrics about pocket selection.

        PT-BR: "n_pockets_total" é o total detectado, "n_pockets_used" é o
        conjunto realmente explorado. O "reduction_ratio" mede a fração de
        redução do espaço de busca (1 - used/total), devendo ser > 0 quando
        total > used.
        """

        reduction_ratio = 0.0
        if total_pockets > 0:
            reduction_ratio = max(0.0, 1.0 - (used_pockets / total_pockets))
        self.log_metric("n_pockets_total", float(total_pockets), step=0)
        self.log_metric("n_pockets_used", float(used_pockets), step=0)
        self.log_metric("reduction_ratio", float(reduction_ratio), step=0)

    def flush(self, out_dir: str) -> None:
        """Write metrics to disk."""

        path = f"{out_dir}/metrics.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")

    def flush_timeseries(self, out_dir: str, mode: str | None = None) -> None:
        """Write per-step metrics to disk."""

        steps: Dict[int, Dict[str, Any]] = {}
        for record in self.records:
            step = record.get("step")
            name = record.get("name")
            value = record.get("value")
            if step is None or name is None:
                continue
            entry = steps.setdefault(int(step), {"step": int(step)})
            entry[name] = value

        if mode:
            for entry in steps.values():
                entry["mode"] = mode

        path = f"{out_dir}/metrics.timeseries.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for step in sorted(steps):
                handle.write(json.dumps(steps[step]) + "\n")
