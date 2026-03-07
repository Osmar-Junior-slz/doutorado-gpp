"""Testes smoke para plots de relatórios (PT-BR)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from dockingpp.reporting.plots import (
    plot_convergencia,
    plot_comparacao_custo,
    plot_comparacao_pareada,
    plot_reducao_espaco_busca,
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

    convergencia_png = tmp_path / "convergence.png"
    cost_png = tmp_path / "cost.png"
    reducao_png = tmp_path / "reduction.png"
    pareado_png = tmp_path / "paired.png"

    assert plot_convergencia(series, convergencia_png)
    assert plot_comparacao_custo(
        {"runtime_total_s": 10.0, "n_eval_total": 300.0},
        {"runtime_total_s": 5.0, "n_eval_total": 120.0},
        cost_png,
    )
    assert plot_reducao_espaco_busca(
        {"n_pockets_total": 20, "n_pockets_used": 5},
        reducao_png,
    )
    assert plot_comparacao_pareada(
        [{"speedup_runtime": 2.0, "speedup_eval": 2.5, "delta_score_cheap": -0.1}],
        pareado_png,
    )

    assert convergencia_png.exists()
    assert cost_png.exists()
    assert reducao_png.exists()
    assert pareado_png.exists()
