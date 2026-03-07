"""Testes smoke para plots de relatórios."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from dockingpp.reporting.plots import (
    plot_confidence,
    plot_convergence,
    plot_cost_vs_quality,
    plot_pockets_total_vs_selected,
    plot_search_reduction,
    plot_trigger_timeline,
)


def test_plots_geram_figuras() -> None:
    assert plot_search_reduction(10, 5, 0.5) is not None
    assert plot_convergence([0, 1, 2], [1.0, 1.2, 1.3]) is not None
    assert plot_cost_vs_quality(100, 10, 1.0, 1.1) is not None
    assert plot_trigger_timeline([1, 3]) is not None
    assert plot_pockets_total_vs_selected(10, 2) is not None
    assert plot_confidence(0.8) is not None
