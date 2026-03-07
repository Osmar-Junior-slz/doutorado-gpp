"""Agregações de métricas para gráficos e tabelas."""

from __future__ import annotations

from typing import Any

from dockingpp.reporting.models import ReportBundle


def compute_search_reduction(bundle: ReportBundle) -> dict[str, float]:
    return {
        "omega_full": bundle.summary.omega_full,
        "omega_reduced": bundle.summary.omega_reduced,
        "omega_ratio": bundle.summary.omega_ratio,
    }


def compute_cost_quality(bundle: ReportBundle) -> dict[str, Any]:
    return {
        "n_evals_cheap": bundle.summary.n_evals_cheap,
        "n_evals_expensive": bundle.summary.n_evals_expensive,
        "best_score_cheap": bundle.summary.best_score_cheap,
        "best_score_expensive": bundle.summary.best_score_expensive,
    }


def compute_pockets(bundle: ReportBundle) -> dict[str, int]:
    return {
        "n_pockets_total": bundle.summary.n_pockets_total,
        "n_pockets_selected": bundle.summary.n_pockets_selected,
    }
