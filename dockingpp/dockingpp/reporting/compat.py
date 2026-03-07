"""Compatibilidade de formatos legados para o schema novo."""

from __future__ import annotations

from typing import Any


LEGACY_STATUS_MAP = {"ok": "success", "error": "failed"}


def normalize_legacy_summary(payload: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    status = payload.get("status") or payload.get("state") or "success"
    status = LEGACY_STATUS_MAP.get(str(status), str(status))
    n_total = payload.get("n_pockets_total", payload.get("n_pockets_detected", 0))
    n_sel = payload.get("n_pockets_selected", payload.get("n_pockets_used", 0))
    omega_full = payload.get("omega_full", n_total)
    omega_reduced = payload.get("omega_reduced", n_sel)
    omega_ratio = payload.get("omega_ratio")
    if omega_ratio is None:
        omega_ratio = (float(omega_reduced) / float(omega_full)) if omega_full else 0.0

    return {
        "run_id": run_id or payload.get("run_id") or "legacy-run",
        "status": status,
        "mode": payload.get("mode", "single"),
        "engine": payload.get("engine", "unknown"),
        "receptor": str(payload.get("receptor", payload.get("receptor_path", "unknown"))),
        "peptide": str(payload.get("peptide", payload.get("peptide_path", "unknown"))),
        "runtime_sec": float(payload.get("runtime_sec", payload.get("total_s", 0.0)) or 0.0),
        "omega_full": float(omega_full or 0.0),
        "omega_reduced": float(omega_reduced or 0.0),
        "omega_ratio": float(omega_ratio or 0.0),
        "n_pockets_total": int(n_total or 0),
        "n_pockets_selected": int(n_sel or 0),
        "n_evals_cheap": int(payload.get("n_evals_cheap", payload.get("n_eval_total", 0)) or 0),
        "n_evals_expensive": int(payload.get("n_evals_expensive", payload.get("expensive_ran_count", 0)) or 0),
        "best_score_cheap": payload.get("best_score_cheap"),
        "best_score_expensive": payload.get("best_score_expensive"),
        "confidence_final": payload.get("confidence_final"),
        "trigger_count_expensive": int(payload.get("trigger_count_expensive", payload.get("expensive_ran_count", 0)) or 0),
        "legacy": payload,
    }


def normalize_legacy_metric(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    event = record.get("event")
    if not event:
        name = record.get("name")
        event = {
            "best_score": "search_iteration",
            "n_eval": "search_iteration",
            "expensive_ran": "expensive_eval",
            "expensive_skipped": "trigger_expensive",
        }.get(str(name), "search_iteration")
    return {
        "event": event,
        "run_id": run_id,
        "step": record.get("step"),
        "pocket_index": record.get("pocket_index"),
        "pocket_id": record.get("pocket_id"),
        "value": record.get("value"),
        "context": record.get("extras", {}),
    }
