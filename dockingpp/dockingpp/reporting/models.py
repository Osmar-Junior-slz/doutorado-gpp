"""Modelos Pydantic para artefatos de execução e relatórios."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RunStatus = Literal["success", "failed", "running"]


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    mode: str
    engine: str
    receptor: str
    peptide: str
    runtime_sec: float = 0.0
    omega_full: float = 0.0
    omega_reduced: float = 0.0
    omega_ratio: float = 0.0
    n_pockets_total: int = 0
    n_pockets_selected: int = 0
    n_evals_cheap: int = 0
    n_evals_expensive: int = 0
    best_score_cheap: float | None = None
    best_score_expensive: float | None = None
    confidence_final: float | None = None
    trigger_count_expensive: int = 0
    error_type: str | None = None
    error_message: str | None = None
    legacy: dict[str, Any] = Field(default_factory=dict)


class MetricEvent(BaseModel):
    event: str
    run_id: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    step: int | None = None
    pocket_id: str | None = None
    pocket_index: int | None = None
    pose_id: str | None = None
    value: float | None = None
    reason: str | None = None
    threshold: float | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ArtifactsManifest(BaseModel):
    run_id: str
    out_dir: str
    files: dict[str, str]
    status: RunStatus


class ReportBundle(BaseModel):
    summary: RunSummary
    metrics: list[MetricEvent] = Field(default_factory=list)
    manifest: ArtifactsManifest | None = None
    warnings: list[str] = Field(default_factory=list)
