"""Reports page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.report_service import (
    build_compare_table,
    find_runs,
    load_json,
    load_jsonl,
    metrics_series,
    summarize_metrics,
)
from dockingpp.gui.state import AppState, StateKeys, set_state
from dockingpp.gui.ui.components import download_json_button


class ReportsPage(BasePage):
    id = "Relatórios"
    title = "Relatórios"

    def render(self, state: AppState) -> None:
        st.header("Relatórios")
        st.session_state.setdefault(StateKeys.REPORTS_ROOT, state.reports_root)
        root_input = st.text_input("Diretório de execuções", value=st.session_state[StateKeys.REPORTS_ROOT])

        if state.last_out_dir:
            st.caption(f"Última execução: {state.last_out_dir}")

        if st.button("Buscar execuções"):
            base_dir = Path(root_input).expanduser()
            if not base_dir.exists():
                st.warning("Diretório informado não existe.")
                set_state(**{StateKeys.REPORT_RUNS: [], StateKeys.REPORTS_ROOT: root_input})
            else:
                runs = find_runs(base_dir)
                set_state(**{StateKeys.REPORT_RUNS: runs, StateKeys.REPORTS_ROOT: root_input})

        runs = st.session_state.get(StateKeys.REPORT_RUNS, [])
        if not runs:
            st.info("Nenhuma execução encontrada ainda. Clique em 'Buscar execuções'.")
            return

        base_dir = Path(st.session_state.get(StateKeys.REPORTS_ROOT, "runs")).expanduser()
        options = [str(run.relative_to(base_dir)) if run.is_relative_to(base_dir) else str(run) for run in runs]
        default_index = 0
        if state.last_out_dir:
            try:
                last_path = Path(state.last_out_dir)
                if last_path in runs:
                    default_index = runs.index(last_path)
            except Exception:  # noqa: BLE001
                default_index = 0

        selected = st.selectbox("Execução", options=options, index=default_index)
        selected_path = runs[options.index(selected)]

        report_path = selected_path / "report.json"
        result_path = selected_path / "result.json"
        metrics_path = selected_path / "metrics.jsonl"

        if report_path.exists():
            try:
                report_data = load_json(report_path)
                rows = build_compare_table(report_data)
                if rows:
                    st.subheader("Comparação Full vs Reduced")
                    st.table(rows)
                else:
                    st.warning("report.json encontrado, mas sem dados de comparação completos.")
            except Exception as exc:  # noqa: BLE001
                st.warning("Não foi possível ler report.json.")
                st.exception(exc)
        else:
            st.warning("report.json não encontrado nesta execução.")

        metrics_records = load_jsonl(metrics_path)
        metrics_summary = summarize_metrics(metrics_records)
        if result_path.exists():
            try:
                result_data = load_json(result_path)
                st.subheader("Resumo da execução")
                summary_rows = [
                    {"Campo": "Melhor score (cheap)", "Valor": result_data.get("best_score_cheap")},
                    {"Campo": "Avaliações", "Valor": metrics_summary.get("n_eval")},
                    {"Campo": "Bolsões usados", "Valor": metrics_summary.get("n_pockets_used")},
                    {"Campo": "Razão de redução", "Valor": metrics_summary.get("reduction_ratio")},
                    {"Campo": "Tempo (s)", "Valor": result_data.get("elapsed_s")},
                ]
                st.table(summary_rows)
            except Exception as exc:  # noqa: BLE001
                st.warning("Não foi possível ler result.json.")
                st.exception(exc)
        else:
            st.warning("result.json não encontrado nesta execução.")

        if metrics_path.exists():
            series, _ = metrics_series(metrics_records, ["best_score_cheap", "best_score", "best"])
            if series:
                st.subheader("Evolução do score")
                st.line_chart(series, x="step", y="score")
            else:
                st.warning("metrics.jsonl encontrado, mas não há dados de score para plotar.")
        else:
            st.warning("metrics.jsonl não encontrado nesta execução.")

        st.write("Downloads")
        download_json_button("Baixar result.json", result_path, filename="result.json", warn_missing=False)
        download_json_button("Baixar report.json", report_path, filename="report.json", warn_missing=False)
        download_json_button("Baixar metrics.jsonl", metrics_path, filename="metrics.jsonl", warn_missing=False)
