"""Testes smoke para plots de relatórios (PT-BR)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from dockingpp.reporting.plots import (
    plot_cost_quality,
    plot_omega_reduction,
    plot_score_stability,
)


def test_plots_geram_pngs(tmp_path: Path) -> None:
    """Gera PNGs básicos para garantir que matplotlib está operacional."""

    series = {
        "iter": [0, 1, 2],
        "best_cheap": [1.0, 1.5, 1.7],
        "best_expensive": [None, 1.6, 1.8],
        "n_eval_total": [10, 20, 30],
        "n_filtered": [2, 4, 6],
        "n_selected": [8, 16, 24],
        "runtime_s": [0.5, 1.0, 1.5],
        "expensive_ran": [0, 1, 1],
    }

    omega_png = tmp_path / "omega.png"
    cost_png = tmp_path / "cost.png"
    stability_png = tmp_path / "stability.png"

    plot_omega_reduction(series, omega_png)
    plot_cost_quality(series, cost_png)
    plot_score_stability(series, stability_png)

    assert omega_png.exists()
    assert cost_png.exists()
    assert stability_png.exists()
