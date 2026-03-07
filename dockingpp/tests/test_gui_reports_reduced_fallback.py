"""Testes para GUI/plots em cenário reduced_aggregate com fallback_to_full."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dockingpp.gui.pages.reports import ReportsPage
from dockingpp.gui.services.report_service import ReportBundle, build_compare_table
from dockingpp.reporting.loaders import find_report_runs
from dockingpp.reporting.plots import plot_cost_comparison


def test_gui_kpi_maps_reduced_aggregate_fields() -> None:
    report_data = {
        "full": {"n_eval_total": 100, "runtime_total_s": 10.0, "best_score_cheap": 1.5},
        "reduced": {
            "mode": "reduced_aggregate",
            "total_n_eval": 42,
            "total_runtime_sec": 3.5,
            "best_over_pockets_cheap": 1.2,
            "best_over_pockets_expensive": 1.1,
            "n_pockets_total": 10,
            "n_pockets_used": 0,
            "fallback_to_full": True,
            "fallback_reason": "no_feasible_pocket",
            "executed_mode": "full",
        },
    }

    rows = build_compare_table(report_data)
    reduced = next(row for row in rows if row["Modo"] == "Reduzido")

    assert reduced["Avaliações"] == 42
    assert reduced["Tempo (s)"] == 3.5
    assert reduced["Melhor score (cheap)"] == 1.2
    assert reduced["Melhor score (expensive)"] == 1.1
    assert reduced["Bolsões usados"] == 0
    assert reduced["fallback_to_full"] is True
    assert reduced["fallback_reason"] == "no_feasible_pocket"
    assert reduced["executed_mode"] == "full"


def test_gui_shows_fallback_warning_when_no_feasible_pocket(monkeypatch) -> None:
    warnings: list[str] = []

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.table", lambda *args, **kwargs: None)
    class _DummyCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.columns", lambda n: [_DummyCol() for _ in range(n)])
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.caption", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.warning", lambda msg, *args, **kwargs: warnings.append(str(msg)))

    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_cost_comparison", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_search_space_reduction", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_convergence", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_paired_comparison", lambda *args, **kwargs: False)
    page = ReportsPage()
    page._render_compare_report(
        ReportBundle(kind="compare", main_json={}, metrics=None, aux_jsons={}),
        metrics_full=None,
        metrics_reduced=None,
        summary_full={"mode": "single"},
        summary_reduced={
            "mode": "reduced_aggregate",
            "fallback_to_full": True,
            "fallback_reason": "no_feasible_pocket",
            "n_pockets_used": 0,
            "per_pocket_results": [],
        },
    )

    assert any("fallback" in msg.lower() for msg in warnings)


def test_plots_hide_reduced_pocket_chart_when_per_pocket_results_empty(monkeypatch) -> None:
    called = {"plot": False}

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.json", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.warning", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.caption", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.dataframe", lambda *args, **kwargs: None)

    def _fake_plot(*args, **kwargs):
        called["plot"] = True

    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_pocket_rank_effect", _fake_plot)

    page = ReportsPage()
    page._render_reduced_aggregate_report(
        ReportBundle(kind="reduced_aggregate", main_json={}, metrics=None, aux_jsons={}),
        metrics_records=None,
        summary_data={"mode": "reduced_aggregate", "per_pocket_results": [], "fallback_to_full": True},
    )

    assert called["plot"] is False


def test_plots_use_compare_fields_for_reduced_fallback_cost(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out_png = tmp_path / "cost.png"
    assert plot_cost_comparison(
        {"runtime_total_s": 10.0, "n_eval_total": 100.0},
        {"reduced_total_runtime_sec": 4.0, "reduced_total_n_eval": 40.0},
        out_png,
    )
    assert out_png.exists()


def test_loaders_normalize_global_and_pockets_aliases(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    reduced_dir = tmp_path / "reduced"
    full_dir.mkdir()
    reduced_dir.mkdir()

    (full_dir / "config.json").write_text(json.dumps({"seed": 1, "generations": 1, "pop_size": 2, "search_space_mode": "global"}), encoding="utf-8")
    (full_dir / "summary.json").write_text(json.dumps({"run_id": "f", "best_cheap_by_pocket": {}, "n_eval_total": 1, "mode": "single"}), encoding="utf-8")

    (reduced_dir / "config.json").write_text(json.dumps({"seed": 1, "generations": 1, "pop_size": 2, "search_space_mode": "pockets"}), encoding="utf-8")
    (reduced_dir / "summary.json").write_text(json.dumps({"run_id": "r", "best_cheap_by_pocket": {}, "n_eval_total": 1, "mode": "single"}), encoding="utf-8")

    runs = find_report_runs(tmp_path)
    kinds = sorted(run.kind for run in runs)
    assert kinds == ["full", "reduced"]


def test_gui_reads_fallback_full_metrics_when_available(tmp_path: Path) -> None:
    fallback_dir = tmp_path / "fallback_full"
    fallback_dir.mkdir()
    metrics = fallback_dir / "metrics.jsonl"
    metrics.write_text(json.dumps({"step": 0, "best_score_cheap": 1.0, "n_eval_total": 5}) + "\n", encoding="utf-8")

    page = ReportsPage()
    records, timeseries = page._load_fallback_metrics_series(
        None,
        {"fallback_to_full": True, "fallback_full_outdir": str(fallback_dir)},
    )

    assert records is not None
    assert records[0]["n_eval_total"] == 5
    assert timeseries is None
