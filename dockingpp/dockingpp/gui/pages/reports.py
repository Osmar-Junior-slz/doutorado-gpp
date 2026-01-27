"""Página de relatórios para visualização de métricas e resultados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.dialog_service import choose_directory
from dockingpp.gui.services.report_service import (
    ReportBundle,
    build_compare_table,
    infer_json_kind,
    load_json,
    load_jsonl,
    metrics_series,
)
from dockingpp.gui.state import AppState, StateKeys
from dockingpp.gui.ui.components import download_json_button


class ReportsPage(BasePage):
    """Página de relatórios com gráficos de convergência e comparações."""

    id = "Relatórios"
    title = "Relatórios"

    @staticmethod
    def _is_monotonic(values: list[float]) -> bool:
        """Verifica se a sequência é monotônica não decrescente."""

        return all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))

    @staticmethod
    def _series_values(series: list[dict[str, float]]) -> list[float]:
        """Extrai os valores numéricos da série, ignorando ausências."""

        return [float(item["score"]) for item in series if item.get("score") is not None]

    @staticmethod
    def _prepare_series(series: list[dict[str, float]]) -> list[dict[str, float]]:
        """Ordena a série pelo step para garantir plotagem consistente."""

        return sorted(series, key=lambda item: item.get("step", 0))

    @staticmethod
    def _compute_evals_cumulative(series: list[dict[str, float]]) -> list[dict[str, float]]:
        """Calcula avaliações acumuladas, preservando séries já cumulativas."""

        ordered = ReportsPage._prepare_series(series)
        values = ReportsPage._series_values(ordered)
        if not values:
            return []
        if ReportsPage._is_monotonic(values):
            # PT-BR: se já é cumulativo, mantemos para evitar dupla soma.
            return [{"step": item["step"], "score": float(item["score"])} for item in ordered]
        cumulative = []
        total = 0.0
        for item in ordered:
            value = float(item["score"])
            total += value
            # PT-BR: somamos incrementalmente quando a série é incremental.
            cumulative.append({"step": item["step"], "score": total})
        return cumulative

    @staticmethod
    def _pair_best_vs_evals(
        best_series: list[dict[str, float]],
        evals_cumulative: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        """Associa best score às avaliações acumuladas pelo mesmo step."""

        best_by_step = {
            item["step"]: float(item["score"]) for item in best_series if item.get("score") is not None
        }
        paired = []
        for item in evals_cumulative:
            step = item.get("step")
            if step in best_by_step and item.get("score") is not None:
                # PT-BR: usamos avaliações no eixo X e best score no eixo Y.
                paired.append({"step": float(item["score"]), "score": best_by_step[step]})
        return paired

    @staticmethod
    def _plot_line(
        data: list[dict[str, float]],
        x_key: str,
        y_key: str,
        title: str,
        xlabel: str,
        ylabel: str,
        label: str | None = None,
    ) -> plt.Figure:
        """Plota uma linha simples a partir de uma lista de dicionários."""

        fig, ax = plt.subplots()
        xs = [item[x_key] for item in data]
        ys = [item[y_key] for item in data]
        ax.plot(xs, ys, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if label:
            ax.legend()
        return fig

    @staticmethod
    def _plot_multi_line(
        series_list: list[list[dict[str, float]]],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> plt.Figure:
        """Plota múltiplas séries em um mesmo gráfico."""

        fig, ax = plt.subplots()
        for series, label in zip(series_list, labels, strict=False):
            xs = [item["step"] for item in series]
            ys = [item["score"] for item in series]
            ax.plot(xs, ys, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return fig

    @staticmethod
    def _plot_scatter(
        xs: list[float],
        ys: list[float],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> plt.Figure:
        """Plota pontos de comparação únicos com rótulos opcionais."""

        fig, ax = plt.subplots()
        for x_val, y_val, label in zip(xs, ys, labels, strict=False):
            ax.scatter([x_val], [y_val], label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if labels:
            ax.legend()
        return fig

    @staticmethod
    def _download_csv_button(label: str, df: pd.DataFrame, filename: str) -> None:
        """Gera um botão de download para CSV a partir de um DataFrame."""

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(label, csv_data, file_name=filename, mime="text/csv")

    @staticmethod
    def _load_uploaded_json(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> dict[str, Any]:
        """Carrega JSON enviado via upload na interface."""

        return json.loads(uploaded_file.getvalue().decode("utf-8"))

    @staticmethod
    def _load_uploaded_jsonl(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> list[dict[str, Any]]:
        """Carrega JSONL enviado via upload na interface."""

        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in content:
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    @staticmethod
    def _download_json_payload(label: str, payload: dict[str, Any], filename: str) -> None:
        """Gera um botão de download para JSON em memória."""

        st.download_button(
            label,
            json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=filename,
            mime="application/json",
        )

    @staticmethod
    def _download_jsonl_payload(label: str, records: list[dict[str, Any]], filename: str) -> None:
        """Gera um botão de download para JSONL em memória."""

        content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        st.download_button(
            label,
            content,
            file_name=filename,
            mime="application/jsonl",
        )

    @staticmethod
    def _summary_value(data: dict[str, Any], keys: list[str]) -> Any:
        """Seleciona o primeiro valor disponível dentre chaves candidatas."""

        return next((data.get(key) for key in keys if data.get(key) is not None), None)

    @staticmethod
    def _guess_metrics_index(options: list[str], tokens: list[str]) -> int:
        """Sugere índice padrão para arquivos de métricas baseado em tokens."""

        for idx, name in enumerate(options, start=1):
            lowered = name.lower()
            if any(token in lowered for token in tokens):
                return idx
        return 0

    def _render_single_report(
        self,
        bundle: ReportBundle,
        metrics_records: list[dict[str, Any]] | None,
    ) -> None:
        """Renderiza relatório de execução única com gráficos e resumos."""

        st.subheader("Resumo da execução")
        summary_rows = [
            {
                "Campo": "Melhor score (cheap)",
                "Valor": self._summary_value(bundle.main_json, ["best_score_cheap", "best_score", "best"]),
            },
            {
                "Campo": "Avaliações",
                "Valor": self._summary_value(bundle.main_json, ["n_eval", "evals", "evaluations"]),
            },
            {"Campo": "Bolsões usados", "Valor": bundle.main_json.get("n_pockets_used")},
            {
                "Campo": "Razão de redução",
                "Valor": self._summary_value(bundle.main_json, ["reduction_ratio", "ratio"]),
            },
            {
                "Campo": "Tempo (s)",
                "Valor": self._summary_value(bundle.main_json, ["elapsed_s", "elapsed_seconds", "elapsed"]),
            },
        ]
        st.table(summary_rows)
        summary_df = pd.DataFrame(summary_rows)
        self._download_csv_button("Baixar tabela resumo (CSV)", summary_df, "resumo_execucao.csv")

        st.subheader("Convergência do score")
        if not metrics_records:
            st.warning("Nenhum arquivo .jsonl selecionado para métricas. Selecione um para ver gráficos.")
            return

        # PT-BR: agregamos por step e aplicamos melhor-so-far para evitar serrilhado.
        best_series, _ = metrics_series(
            metrics_records,
            ["best_score_cheap", "best_score", "best"],
            aggregate="min",
            cumulative_best=True,
        )
        if best_series:
            best_series = self._prepare_series(best_series)
            fig = self._plot_line(
                best_series,
                "step",
                "score",
                "Best score vs geração",
                "Geração",
                "Best score",
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Não foi possível encontrar best score nas métricas.")

        eval_series, _ = metrics_series(metrics_records, ["n_eval", "evals", "evaluations"])
        if eval_series and best_series:
            evals_cum = self._compute_evals_cumulative(eval_series)
            paired_series = self._pair_best_vs_evals(best_series, evals_cum)
            if paired_series:
                fig = self._plot_line(
                    paired_series,
                    "step",
                    "score",
                    "Best score vs avaliações acumuladas",
                    "Avaliações acumuladas",
                    "Best score",
                )
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("Não foi possível calcular avaliações acumuladas.")
        elif not eval_series:
            st.warning("n_eval não disponível para plotar avaliações.")

    def _render_compare_report(
        self,
        bundle: ReportBundle,
        metrics_full: list[dict[str, Any]] | None,
        metrics_reduced: list[dict[str, Any]] | None,
    ) -> None:
        """Renderiza relatório comparativo entre modos completo e reduzido."""

        st.subheader("Comparação: Completo vs Reduzido")
        rows = build_compare_table(bundle.main_json)
        if rows:
            st.table(rows)
            compare_df = pd.DataFrame(rows)
            self._download_csv_button("Baixar tabela resumo (CSV)", compare_df, "comparacao_full_reduced.csv")
        else:
            st.warning("Não foi possível montar a tabela de comparação com os dados disponíveis.")

        if not metrics_full and not metrics_reduced:
            st.warning("Nenhum arquivo .jsonl selecionado para métricas de comparação.")
            return
        if not metrics_full or not metrics_reduced:
            st.warning("Selecione arquivos .jsonl para completo e reduzido para comparar curvas.")
            return

        st.subheader("Convergência (Full vs Reduced)")
        full_best, _ = metrics_series(
            metrics_full,
            ["best_score_cheap", "best_score", "best"],
            aggregate="min",
            cumulative_best=True,
        )
        reduced_best, _ = metrics_series(
            metrics_reduced,
            ["best_score_cheap", "best_score", "best"],
            aggregate="min",
            cumulative_best=True,
        )
        if full_best and reduced_best:
            full_best = self._prepare_series(full_best)
            reduced_best = self._prepare_series(reduced_best)
            fig = self._plot_multi_line(
                [full_best, reduced_best],
                ["Completo", "Reduzido"],
                "Best score vs geração",
                "Geração",
                "Best score",
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Não foi possível montar séries de best score para comparação.")

        full_eval, _ = metrics_series(metrics_full, ["n_eval", "evals", "evaluations"])
        reduced_eval, _ = metrics_series(metrics_reduced, ["n_eval", "evals", "evaluations"])
        if full_eval and reduced_eval and full_best and reduced_best:
            full_eval_cum = self._compute_evals_cumulative(full_eval)
            reduced_eval_cum = self._compute_evals_cumulative(reduced_eval)
            full_best_eval = self._pair_best_vs_evals(full_best, full_eval_cum)
            reduced_best_eval = self._pair_best_vs_evals(reduced_best, reduced_eval_cum)
            fig = self._plot_multi_line(
                [full_best_eval, reduced_best_eval],
                ["Completo", "Reduzido"],
                "Best score vs avaliações acumuladas",
                "Avaliações acumuladas",
                "Best score",
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("n_eval ou best score não disponíveis para comparação por avaliações.")

    def render(self, state: AppState) -> None:
        """Renderiza a página completa de relatórios."""

        st.header("Relatórios")
        if st.session_state.get(StateKeys.REPORTS_ROOT_PENDING):
            st.session_state[StateKeys.REPORTS_ROOT] = st.session_state.pop(StateKeys.REPORTS_ROOT_PENDING)
        if StateKeys.REPORTS_ROOT not in st.session_state:
            st.session_state[StateKeys.REPORTS_ROOT] = "runs"

        source = st.radio(
            "Fonte do relatório",
            ["Selecionar pasta", "Selecionar arquivo JSON"],
            horizontal=True,
        )

        main_json: dict[str, Any] | None = None
        aux_jsons: dict[str, dict[str, Any]] = {}
        metrics_records: list[dict[str, Any]] | None = None
        metrics_full: list[dict[str, Any]] | None = None
        metrics_reduced: list[dict[str, Any]] | None = None
        main_json_path: Path | None = None
        metrics_path: Path | None = None
        metrics_full_path: Path | None = None
        metrics_reduced_path: Path | None = None
        main_json_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        metrics_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        metrics_full_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        metrics_reduced_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None

        if source == "Selecionar pasta":
            folder_path = st.text_input("Pasta do relatório", key=StateKeys.REPORTS_ROOT)
            if st.button("Selecionar pasta..."):
                chosen = choose_directory()
                if chosen is not None:
                    st.session_state[StateKeys.REPORTS_ROOT_PENDING] = chosen
                    st.rerun()
            if not folder_path:
                st.info("Informe uma pasta com arquivos JSON para continuar.")
                return

            selected_folder = Path(folder_path).expanduser()
            if not selected_folder.exists():
                st.warning("Pasta informada não existe.")
                return

            json_paths = sorted(selected_folder.glob("*.json"))
            if not json_paths:
                st.warning("Nenhum arquivo .json encontrado na pasta.")
                return

            json_map = {path.name: path for path in json_paths}
            selected_name = st.selectbox("Escolha o JSON principal", list(json_map.keys()))
            main_json_path = json_map[selected_name]
            json_payloads = {name: load_json(path) for name, path in json_map.items()}
            main_json = json_payloads[selected_name]
            aux_jsons = {name: payload for name, payload in json_payloads.items() if name != selected_name}

            jsonl_paths = sorted(selected_folder.glob("*.jsonl"))
            jsonl_map = {path.name: path for path in jsonl_paths}

            kind = infer_json_kind(main_json)
            if kind == "single" and jsonl_map:
                options = ["Nenhum"] + list(jsonl_map.keys())
                selected_metrics = st.selectbox("Arquivo de métricas (.jsonl)", options)
                if selected_metrics != "Nenhum":
                    metrics_path = jsonl_map[selected_metrics]
                    metrics_records = load_jsonl(metrics_path)
            elif kind == "compare" and jsonl_map:
                options = ["Nenhum"] + list(jsonl_map.keys())
                default_full = self._guess_metrics_index(list(jsonl_map.keys()), ["full", "completo"])
                default_reduced = self._guess_metrics_index(list(jsonl_map.keys()), ["reduced", "reduzido"])
                selected_full = st.selectbox(
                    "Métricas do modo completo (.jsonl)",
                    options,
                    index=default_full,
                    key="metrics_full_folder",
                )
                selected_reduced = st.selectbox(
                    "Métricas do modo reduzido (.jsonl)",
                    options,
                    index=default_reduced,
                    key="metrics_reduced_folder",
                )
                if selected_full != "Nenhum":
                    metrics_full_path = jsonl_map[selected_full]
                    metrics_full = load_jsonl(metrics_full_path)
                if selected_reduced != "Nenhum":
                    metrics_reduced_path = jsonl_map[selected_reduced]
                    metrics_reduced = load_jsonl(metrics_reduced_path)
        else:
            main_json_upload = st.file_uploader("JSON principal", type=["json"])
            if not main_json_upload:
                st.info("Selecione um arquivo JSON para continuar.")
                return
            main_json = self._load_uploaded_json(main_json_upload)
            kind = infer_json_kind(main_json)

            if kind == "single":
                metrics_upload = st.file_uploader("Arquivo de métricas (.jsonl)", type=["jsonl"])
                if metrics_upload:
                    metrics_records = self._load_uploaded_jsonl(metrics_upload)
            elif kind == "compare":
                metrics_full_upload = st.file_uploader(
                    "Métricas do modo completo (.jsonl)",
                    type=["jsonl"],
                    key="metrics_full_upload",
                )
                metrics_reduced_upload = st.file_uploader(
                    "Métricas do modo reduzido (.jsonl)",
                    type=["jsonl"],
                    key="metrics_reduced_upload",
                )
                if metrics_full_upload:
                    metrics_full = self._load_uploaded_jsonl(metrics_full_upload)
                if metrics_reduced_upload:
                    metrics_reduced = self._load_uploaded_jsonl(metrics_reduced_upload)

        if main_json is None:
            return

        bundle = ReportBundle(
            kind=infer_json_kind(main_json),
            main_json=main_json,
            metrics=metrics_records,
            aux_jsons=aux_jsons,
        )

        if bundle.kind == "single":
            self._render_single_report(bundle, metrics_records)
        elif bundle.kind == "compare":
            self._render_compare_report(bundle, metrics_full, metrics_reduced)
        else:
            st.warning("Não foi possível identificar o tipo do JSON. Exibindo conteúdo bruto.")
            st.json(bundle.main_json)

        st.subheader("Downloads")
        if main_json_path:
            download_json_button("Baixar JSON principal", main_json_path, filename=main_json_path.name, warn_missing=False)
        elif main_json_upload:
            self._download_json_payload("Baixar JSON principal", main_json, main_json_upload.name)

        if bundle.kind == "single":
            if metrics_path:
                st.download_button(
                    "Baixar métricas (.jsonl)",
                    data=metrics_path.read_text(encoding="utf-8"),
                    file_name=metrics_path.name,
                    mime="application/jsonl",
                )
            elif metrics_upload and metrics_records is not None:
                self._download_jsonl_payload("Baixar métricas (.jsonl)", metrics_records, metrics_upload.name)
        elif bundle.kind == "compare":
            if metrics_full_path:
                st.download_button(
                    "Baixar métricas completo (.jsonl)",
                    data=metrics_full_path.read_text(encoding="utf-8"),
                    file_name=metrics_full_path.name,
                    mime="application/jsonl",
                )
            elif metrics_full_upload and metrics_full is not None:
                self._download_jsonl_payload(
                    "Baixar métricas completo (.jsonl)",
                    metrics_full,
                    metrics_full_upload.name,
                )

            if metrics_reduced_path:
                st.download_button(
                    "Baixar métricas reduzido (.jsonl)",
                    data=metrics_reduced_path.read_text(encoding="utf-8"),
                    file_name=metrics_reduced_path.name,
                    mime="application/jsonl",
                )
            elif metrics_reduced_upload and metrics_reduced is not None:
                self._download_jsonl_payload(
                    "Baixar métricas reduzido (.jsonl)",
                    metrics_reduced,
                    metrics_reduced_upload.name,
                )
