"""Testes smoke para plots de relatórios (PT-BR)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from dockingpp.reporting.plots import (
    plot_convergence,
    plot_cost_comparison,
    plot_paired_comparison,
    plot_search_space_reduction,
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

    convergence_png = tmp_path / "convergence.png"
    cost_png = tmp_path / "cost.png"
    reduction_png = tmp_path / "reduction.png"
    paired_png = tmp_path / "paired.png"

    assert plot_convergence(series, convergence_png)
    assert plot_cost_comparison(
        {"runtime_total_s": 10.0, "n_eval_total": 300.0},
        {"runtime_total_s": 5.0, "n_eval_total": 120.0},
        cost_png,
    )
    assert plot_search_space_reduction(
        {"n_pockets_total": 20, "n_pockets_used": 5},
        reduction_png,
    )
    assert plot_paired_comparison(
        [{"speedup_runtime": 2.0, "speedup_eval": 2.5, "delta_score_cheap": -0.1}],
        paired_png,
    )

    assert convergence_png.exists()
    assert cost_png.exists()
    assert reduction_png.exists()
    assert paired_png.exists()
