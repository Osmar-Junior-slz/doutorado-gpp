"""Testes para GUI/plots em cenário reduced_aggregate com fallback_to_full."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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

    assert reduced["Avalia\u00e7\u00f5es"] == 42
    assert reduced["Tempo (s)"] == 3.5
    assert reduced["Melhor score (cheap)"] == 1.2
    assert reduced["Melhor score (expensive)"] == 1.1
    assert reduced["Bols\u00f5es usados"] == 0
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



def test_compare_shows_reduced_per_pocket_rows_from_summary(monkeypatch) -> None:
    table_calls: list[object] = []
    dataframe_calls: list[pd.DataFrame] = []
    pocket_payloads: list[dict[str, object]] = []

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.table", lambda obj, *args, **kwargs: table_calls.append(obj))
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.warning", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.caption", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.dataframe", lambda obj, *args, **kwargs: dataframe_calls.append(obj.copy()))

    class _DummyCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.columns", lambda n: [_DummyCol() for _ in range(n)])
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_cost_comparison", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_search_space_reduction", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_convergence", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_paired_comparison", lambda *args, **kwargs: False)

    def _fake_pocket_plot(payload, *_args, **_kwargs):
        pocket_payloads.append(payload)
        return False

    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_pocket_rank_effect", _fake_pocket_plot)

    page = ReportsPage()
    page._render_compare_report(
        ReportBundle(
            kind="compare",
            main_json={
                "full": {"best_score_cheap": 1.5, "n_eval_total": 100, "runtime_total_s": 10.0},
                "reduced": {"mode": "reduced_aggregate", "best_over_pockets_cheap": 1.2, "total_n_eval": 40, "total_runtime_sec": 4.0},
            },
            metrics=None,
            aux_jsons={},
        ),
        metrics_full=None,
        metrics_reduced=None,
        summary_full={"mode": "single"},
        summary_reduced={
            "mode": "reduced_aggregate",
            "fallback_to_full": False,
            "best_pocket_id": "p2",
            "per_pocket_results": [
                {"pocket_id": "p1", "pocket_index": 0, "best_score_cheap": 1.0, "best_score_expensive": 0.7, "runtime_sec": 2.0, "n_eval_total": 20},
                {"pocket_id": "p2", "pocket_index": 1, "best_score_cheap": 1.2, "best_score_expensive": 0.8, "runtime_sec": 2.5, "n_eval_total": 20},
            ],
        },
    )

    assert any(isinstance(obj, pd.DataFrame) and "Modo" in obj.columns for obj in table_calls)
    assert len(dataframe_calls) == 1
    df = dataframe_calls[0]
    assert list(df["pocket_id"]) == ["p1", "p2"]
    assert {"pocket_rank", "best_score_cheap", "best_score_expensive", "runtime_sec", "n_eval"} <= set(df.columns)
    assert pocket_payloads
    assert len(pocket_payloads[0]["per_pocket_results"]) == 2


def test_compare_uses_subfolder_fallback_when_summary_has_no_per_pocket(monkeypatch, tmp_path: Path) -> None:
    dataframe_calls: list[pd.DataFrame] = []
    captions: list[str] = []

    reduced_dir = tmp_path / "reduced_run"
    reduced_dir.mkdir()
    p1_dir = reduced_dir / "p1"
    p2_dir = reduced_dir / "p2"
    p1_dir.mkdir()
    p2_dir.mkdir()
    (p1_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "single",
                "search_space_mode": "reduced",
                "best_pose_pocket_id": "p1",
                "best_score_cheap": 1.0,
                "best_score_expensive": 0.7,
                "n_eval_total": 15,
            }
        ),
        encoding="utf-8",
    )
    (p2_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "single",
                "search_space_mode": "reduced",
                "best_pose_pocket_id": "p2",
                "best_score_cheap": 1.1,
                "best_score_expensive": 0.8,
                "n_eval_total": 12,
            }
        ),
        encoding="utf-8",
    )
    (p1_dir / "result.json").write_text(json.dumps({"mode": "single", "runtime_sec": 1.5}), encoding="utf-8")
    (p2_dir / "result.json").write_text(json.dumps({"mode": "single", "runtime_sec": 1.1}), encoding="utf-8")

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.table", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.warning", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.caption", lambda msg, *args, **kwargs: captions.append(str(msg)))
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.dataframe", lambda obj, *args, **kwargs: dataframe_calls.append(obj.copy()))

    class _DummyCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.columns", lambda n: [_DummyCol() for _ in range(n)])
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_cost_comparison", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_search_space_reduction", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_convergence", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_paired_comparison", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_pocket_rank_effect", lambda *args, **kwargs: False)

    page = ReportsPage()
    page._render_compare_report(
        ReportBundle(kind="compare", main_json={}, metrics=None, aux_jsons={}),
        metrics_full=None,
        metrics_reduced=None,
        summary_full={"mode": "single"},
        summary_reduced={
            "mode": "reduced_aggregate",
            "fallback_to_full": False,
            "per_pocket_results": [],
            "selected_pockets": ["p1", "p2"],
        },
        reduced_run_dir=reduced_dir,
    )

    assert len(dataframe_calls) == 1
    df = dataframe_calls[0]
    assert set(df["pocket_id"]) == {"p1", "p2"}
    assert any("subpastas" in msg for msg in captions)


def test_compare_keeps_fallback_warning_without_inventing_pockets(monkeypatch, tmp_path: Path) -> None:
    warnings: list[str] = []
    dataframes: list[pd.DataFrame] = []

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.table", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.caption", lambda *args, **kwargs: None)
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.warning", lambda msg, *args, **kwargs: warnings.append(str(msg)))
    monkeypatch.setattr("dockingpp.gui.pages.reports.st.dataframe", lambda obj, *args, **kwargs: dataframes.append(obj.copy()))

    class _DummyCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("dockingpp.gui.pages.reports.st.columns", lambda n: [_DummyCol() for _ in range(n)])
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_cost_comparison", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_search_space_reduction", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_convergence", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_paired_comparison", lambda *args, **kwargs: False)
    monkeypatch.setattr("dockingpp.gui.pages.reports.plot_pocket_rank_effect", lambda *args, **kwargs: False)

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
        reduced_run_dir=tmp_path,
    )

    assert any("fallback" in msg.lower() for msg in warnings)
    assert not dataframes

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
