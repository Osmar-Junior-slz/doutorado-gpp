"""Utilitários de logging para o dockingpp."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunLogger:
    """Logger em memória com opção de escrita incremental no JSONL."""

    records: List[Dict[str, Any]] = field(default_factory=list)
    out_dir: str | None = None
    live_write: bool = False

    def log_metric(self, name: str, value: float, step: int, extra: Optional[Dict[str, Any]] = None) -> None:
        extras: Dict[str, Any] = dict(extra or {})
        generation = extras.get("generation")
        pocket_index = extras.get("pocket_index")
        payload: Dict[str, Any] = {
            "step": int(step),
            "pocket_index": int(pocket_index) if pocket_index is not None else None,
            "generation": int(generation) if generation is not None else None,
            "name": name,
            "value": float(value) if value is not None else None,
        }
        if extras:
            payload["extras"] = extras
            for key, val in extras.items():
                if key not in payload:
                    payload[key] = val
        self.records.append(payload)
        self._append_record(payload)

    def _append_record(self, payload: Dict[str, Any]) -> None:
        if not self.live_write or not self.out_dir:
            return
        path = f"{self.out_dir}/metrics.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def log_global_metrics(self, total_pockets: int, used_pockets: int) -> None:
        reduction_ratio = 0.0
        if total_pockets > 0:
            reduction_ratio = max(0.0, 1.0 - (used_pockets / total_pockets))
        self.log_metric("n_pockets_total", float(total_pockets), step=0)
        self.log_metric("n_pockets_used", float(used_pockets), step=0)
        self.log_metric("reduction_ratio", float(reduction_ratio), step=0)

    def flush(self, out_dir: str) -> None:
        path = f"{out_dir}/metrics.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")

    def flush_timeseries(self, out_dir: str, mode: str | None = None) -> None:
        steps: Dict[int, Dict[str, Any]] = {}
        cumulative_eval = 0.0
        cumulative_expensive = 0.0
        for record in self.records:
            step = record.get("step")
            name = record.get("name")
            value = record.get("value")
            if step is None or name is None:
                continue
            step_i = int(step)
            entry = steps.setdefault(step_i, {"step": step_i})
            entry[name] = value
            if name == "best_score":
                entry["best_score_cheap"] = value
            generation = record.get("generation")
            if generation is not None:
                entry["generation"] = int(generation)
            if name == "n_eval" and value is not None:
                cumulative_eval += float(value)
            if name == "expensive_ran" and value is not None:
                cumulative_expensive += float(value)
            entry["n_eval_cumulative"] = cumulative_eval
            entry["expensive_ran_cumulative"] = cumulative_expensive

        if mode:
            for entry in steps.values():
                entry["mode"] = mode

        path = f"{out_dir}/metrics.timeseries.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for step in sorted(steps):
                handle.write(json.dumps(steps[step]) + "\n")


class AuditTracer:
    """Tracer JSONL estruturado centralizado para auditoria de execução."""

    SCHEMA_VERSION = "1.0"
    LEVELS = {"OFF": 0, "BASIC": 1, "TRACE": 2, "AUDIT": 3}

    def __init__(
        self,
        out_dir: str,
        run_id: str,
        debug_enabled: bool = False,
        debug_level: str = "AUDIT",
        debug_dirname: str = "debug",
        search_space_mode: str | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.run_id = run_id
        self.debug_dirname = debug_dirname
        self.debug_level = (debug_level or "AUDIT").upper()
        self.level_value = self.LEVELS.get(self.debug_level, self.LEVELS["AUDIT"])
        self.enabled = bool(debug_enabled) and self.level_value > 0
        self.search_space_mode = search_space_mode
        self.debug_dir = os.path.join(out_dir, debug_dirname)
        self._start_monotonic = time.perf_counter()
        self._start_iso = datetime.now(timezone.utc).isoformat()
        self._event_seq = 0
        self._lock = threading.Lock()
        self._files: dict[str, Any] = {}
        self._children: list["AuditTracer"] = []
        self._generated_files: list[str] = []
        self.warnings_count = 0
        self.errors_count = 0
        if self.enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            self._open_core_files()

    def _open_core_files(self) -> None:
        for rel_path in (
            "trace.jsonl",
            "decision_trace.jsonl",
            "warnings.jsonl",
            "errors.jsonl",
        ):
            self._open_file(rel_path)

    def _open_file(self, rel_path: str):
        full_path = os.path.join(self.debug_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        handle = open(full_path, "a", encoding="utf-8")
        self._files[rel_path] = handle
        self._mark_file(rel_path)
        return handle

    def _mark_file(self, rel_path: str) -> None:
        rel = str(Path(self.debug_dirname) / rel_path)
        if rel not in self._generated_files:
            self._generated_files.append(rel)

    def is_enabled(self, level: str = "BASIC") -> bool:
        return self.enabled and self.level_value >= self.LEVELS.get(level.upper(), 1)

    def start_run(self, payload: Optional[dict[str, Any]] = None) -> None:
        self.event(
            stage="cli",
            event_type="run_started",
            substage="start",
            payload=payload or {},
            level="BASIC",
        )

    def event(
        self,
        *,
        stage: str,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        substage: Optional[str] = None,
        level: str = "TRACE",
        parent_event_id: Optional[str] = None,
        pocket_id: Optional[str] = None,
        module: Optional[str] = None,
        function: Optional[str] = None,
        engine: Optional[str] = None,
        decision: bool = False,
    ) -> Optional[str]:
        if not self.is_enabled(level):
            return None
        with self._lock:
            self._event_seq += 1
            event_id = f"evt-{self._event_seq:08d}"
        event = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "event_id": event_id,
            "parent_event_id": parent_event_id,
            "stage": stage,
            "substage": substage,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_ms": int((time.perf_counter() - self._start_monotonic) * 1000),
            "level": level,
            "module": module,
            "function": function,
            "search_space_mode": self.search_space_mode,
            "pocket_id": pocket_id,
            "engine": engine,
            "payload": payload or {},
        }
        self._write_jsonl("trace.jsonl", event)
        if decision:
            self._write_jsonl("decision_trace.jsonl", event)
        return event_id

    def warning(self, *, stage: str, message: str, payload: Optional[dict[str, Any]] = None, pocket_id: Optional[str] = None) -> None:
        self.warnings_count += 1
        event_id = self.event(
            stage=stage,
            event_type="warning",
            payload={"message": message, **(payload or {})},
            pocket_id=pocket_id,
            level="BASIC",
        )
        if event_id:
            self._write_jsonl("warnings.jsonl", {"event_id": event_id, "message": message, "payload": payload or {}})

    def error(self, *, stage: str, message: str, payload: Optional[dict[str, Any]] = None, pocket_id: Optional[str] = None) -> None:
        self.errors_count += 1
        event_id = self.event(
            stage=stage,
            event_type="run_failed",
            payload={"message": message, **(payload or {})},
            pocket_id=pocket_id,
            level="BASIC",
        )
        if event_id:
            self._write_jsonl("errors.jsonl", {"event_id": event_id, "message": message, "payload": payload or {}})

    def child(self, pocket_id: str) -> "AuditTracer":
        child = AuditTracer(
            out_dir=self.out_dir,
            run_id=self.run_id,
            debug_enabled=self.enabled,
            debug_level=self.debug_level,
            debug_dirname=self.debug_dirname,
            search_space_mode=self.search_space_mode,
        )
        child.debug_dir = self.debug_dir
        child._files = self._files
        child._generated_files = self._generated_files
        child._start_monotonic = self._start_monotonic
        child._start_iso = self._start_iso

        def _event_with_pocket(**kwargs: Any) -> Optional[str]:
            kwargs.setdefault("pocket_id", pocket_id)
            kwargs.setdefault("module", kwargs.get("module"))
            return self.event(**kwargs)

        child.event = _event_with_pocket  # type: ignore[assignment]
        child.warning = lambda **kwargs: self.warning(pocket_id=pocket_id, **kwargs)  # type: ignore[assignment]
        child.error = lambda **kwargs: self.error(pocket_id=pocket_id, **kwargs)  # type: ignore[assignment]
        self._children.append(child)
        return child

    def write_summary(self, summary: dict[str, Any], rel_path: str = "debug_summary.json") -> None:
        if not self.enabled:
            return
        self._write_json(rel_path, summary)

    def finish_run(self, manifest: dict[str, Any], status_final: str) -> None:
        if not self.enabled:
            return
        self.event(stage="summary", event_type="run_finished", payload={"status_final": status_final}, level="BASIC")
        manifest_payload = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            **manifest,
            "timestamp_start": manifest.get("timestamp_start", self._start_iso),
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
            "status_final": status_final,
            "files": sorted(self._generated_files),
        }
        self._write_json("manifest.json", manifest_payload)
        for handle in list(self._files.values()):
            try:
                handle.flush()
            except Exception:
                pass
        for handle in list(self._files.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._files.clear()

    def _write_jsonl(self, rel_path: str, payload: dict[str, Any]) -> None:
        if rel_path not in self._files:
            self._open_file(rel_path)
        handle = self._files[rel_path]
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_json(self, rel_path: str, payload: dict[str, Any]) -> None:
        path = os.path.join(self.debug_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        self._mark_file(rel_path)

    def artifact_written(self, path: str, stage: str = "serialization") -> None:
        self.event(stage=stage, event_type="artifact_written", payload={"path": path}, level="TRACE")

    @property
    def generated_files(self) -> list[str]:
        return sorted(self._generated_files)
