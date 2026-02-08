"""Página de relatórios para visualização de métricas e resultados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import tempfile

import pandas as pd
import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.dialog_service import choose_directory
from dockingpp.gui.services.report_service import (
    ReportBundle,
    build_compare_table,
    infer_json_kind,
)
from dockingpp.gui.state import AppState, StateKeys
from dockingpp.gui.ui.components import download_json_button
from dockingpp.reporting.loaders import extract_series, find_matching_jsonl, load_any_json, load_jsonl
from dockingpp.reporting.plots import (
    plot_cost_quality,
    plot_filter_distribution,
    plot_omega_reduction,
    plot_pocket_rank_effect,
    plot_score_stability,
)


class ReportsPage(BasePage):
    """Página de relatórios com gráficos de convergência e comparações."""

    id = "Relatórios"
    title = "Relatórios"

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
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
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
    def _resolve_report_path(base_dir: Path | None, raw_path: str | None) -> Path | None:
        """Resolve caminhos relativos usando o diretório base do relatório."""

        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate
        if base_dir:
            return base_dir / candidate
        return candidate

    def _build_summary_rows(
        self,
        summary_data: dict[str, Any],
        fallback_scores: dict[str, float | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Monta linhas para a tabela resumo a partir do summary.json."""

        fallback_scores = fallback_scores or {}
        return [
            {
                "Campo": "pockets_detected",
                "Valor": self._summary_value(
                    summary_data,
                    ["pockets_detected", "n_pockets_total", "pockets_total", "n_pockets_detected"],
                ),
            },
            {
                "Campo": "pockets_used",
                "Valor": self._summary_value(
                    summary_data,
                    ["pockets_used", "n_pockets_used"],
                ),
            },
            {
                "Campo": "best_cheap",
                "Valor": self._summary_value(
                    summary_data,
                    ["best_score_cheap", "best_score", "best"],
                )
                or fallback_scores.get("best_cheap"),
            },
            {
                "Campo": "best_expensive",
                "Valor": self._summary_value(
                    summary_data,
                    ["best_score_expensive"],
                )
                or fallback_scores.get("best_expensive"),
            },
            {
                "Campo": "total_s",
                "Valor": self._summary_value(
                    summary_data,
                    ["total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
                ),
            },
            {
                "Campo": "expensive_ran_count",
                "Valor": self._summary_value(
                    summary_data,
                    ["expensive_ran_count"],
                ),
            },
        ]

    @staticmethod
    def _guess_metrics_index(options: list[str], tokens: list[str]) -> int:
        """Sugere índice padrão para arquivos de métricas baseado em tokens."""

        for idx, name in enumerate(options, start=1):
            lowered = name.lower()
            if any(token in lowered for token in tokens):
                return idx
        return 0

    @staticmethod
    def _extract_compare_block(report_data: dict[str, Any], label: str) -> dict[str, Any] | None:
        """Busca o bloco de comparação correspondente ao rótulo fornecido."""

        for container in (report_data, report_data.get("runs"), report_data.get("comparison")):
            if isinstance(container, dict) and label in container and isinstance(container[label], dict):
                return container[label]
        return None

    def _load_summary_data(self, summary_path: Path | None, fallback: dict[str, Any]) -> dict[str, Any]:
        """Carrega summary.json, caindo no fallback em caso de erro."""

        if summary_path and summary_path.exists():
            try:
                return load_any_json(summary_path)
            except (OSError, json.JSONDecodeError):
                return fallback
        return fallback

    @staticmethod
    def _buscar_jsonl_na_pasta(base_dir: Path) -> list[Path]:
        """Lista JSONL disponíveis na pasta e subpastas."""

        candidatos = sorted(base_dir.glob("*.jsonl"))
        candidatos += sorted(base_dir.rglob("metrics.jsonl"))
        vistos: set[Path] = set()
        unicos: list[Path] = []
        for candidato in candidatos:
            if candidato in vistos:
                continue
            vistos.add(candidato)
            unicos.append(candidato)
        return unicos

    @staticmethod
    def _ultimo_valor(lista: list[Any]) -> Any:
        """Retorna o último valor não nulo da lista."""

        for valor in reversed(lista):
            if valor is not None:
                return valor
        return None

    @staticmethod
    def _resumo_gap(series: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        """Monta resumo do gap com base em séries e summary."""

        n_eval = ReportsPage._ultimo_valor(series.get("n_eval_total", []))
        n_filtered = ReportsPage._ultimo_valor(series.get("n_filtered", []))
        n_selected = ReportsPage._ultimo_valor(series.get("n_selected", []))
        runtime = ReportsPage._ultimo_valor(series.get("runtime_s", []))
        expensive = ReportsPage._ultimo_valor(series.get("expensive_ran", []))
        best_cheap = ReportsPage._ultimo_valor(series.get("best_cheap", []))
        best_expensive = ReportsPage._ultimo_valor(series.get("best_expensive", []))
        if runtime is None:
            runtime = ReportsPage._summary_value(
                summary,
                ["total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
            )
        if best_cheap is None:
            best_cheap = ReportsPage._summary_value(summary, ["best_score_cheap", "best_score", "best"])
        if best_expensive is None:
            best_expensive = ReportsPage._summary_value(summary, ["best_score_expensive", "best_expensive"])
        kept_ratio = None
        if n_eval is not None and n_selected is not None:
            kept_ratio = float(n_selected) / max(float(n_eval), 1.0)
        return {
            "n_eval_total": n_eval,
            "n_filtered": n_filtered,
            "kept_ratio_final": kept_ratio,
            "runtime_total_s": runtime,
            "expensive_ran": expensive,
            "best_cheap": best_cheap,
            "best_expensive": best_expensive,
        }

    @staticmethod
    def _render_png(path: Path, legenda: str) -> None:
        """Exibe um PNG no Streamlit."""

        if path.exists():
            st.image(str(path), caption=legenda, use_container_width=True)

    def _render_warning_missing(self, series: dict[str, Any]) -> None:
        """Mostra aviso resumido sobre chaves ausentes."""

        faltas = series.get("missing")
        if not isinstance(faltas, dict):
            return
        faltantes = {chave: count for chave, count in faltas.items() if count}
        if faltantes:
            st.caption(f"Campos ausentes em parte das métricas: {faltantes}")

    def _render_single_report(
        self,
        bundle: ReportBundle,
        metrics_records: list[dict[str, Any]] | None,
        summary_data: dict[str, Any],
    ) -> None:
        """Renderiza relatório de execução única com gráficos e resumos."""

        if not metrics_records:
            st.warning("Nenhum arquivo .jsonl selecionado para métricas. Selecione um para ver gráficos.")
            return

        series = extract_series(metrics_records)
        st.subheader("Resumo do Gap")
        resumo = self._resumo_gap(series, summary_data)
        st.table(pd.DataFrame([resumo]))

        st.subheader("Resumo da execução")
        summary_rows = self._build_summary_rows(summary_data, resumo)
        st.table(summary_rows)
        summary_df = pd.DataFrame(summary_rows)
        self._download_csv_button("Baixar tabela resumo (CSV)", summary_df, "resumo_execucao.csv")

        self._render_warning_missing(series)
        st.subheader("Gráficos")
        temp_dir = Path(tempfile.mkdtemp(prefix="reports_"))

        omega_png = temp_dir / "omega_reduction.png"
        plot_omega_reduction(series, omega_png)
        self._render_png(omega_png, "Redução de Ω para Ω'")

        cost_png = temp_dir / "cost_quality.png"
        plot_cost_quality(series, cost_png)
        self._render_png(cost_png, "Custo x Qualidade")

        stability_png = temp_dir / "score_stability.png"
        plot_score_stability(series, stability_png)
        self._render_png(stability_png, "Estabilidade do score")

        pocket_png = temp_dir / "pocket_effect.png"
        plot_pocket_rank_effect(metrics_records, pocket_png)
        self._render_png(pocket_png, "Efeito do pocket ranking")

        filter_png = temp_dir / "filter_distribution.png"
        if plot_filter_distribution(metrics_records, filter_png):
            self._render_png(filter_png, "Distribuição pré vs pós filtro")
        else:
            st.warning("Dados de distribuição não disponíveis.")

    def _render_compare_report(
        self,
        bundle: ReportBundle,
        metrics_full: list[dict[str, Any]] | None,
        metrics_reduced: list[dict[str, Any]] | None,
        summary_full: dict[str, Any],
        summary_reduced: dict[str, Any],
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

        st.subheader("Resumo do summary.json")
        series_full = extract_series(metrics_full or [])
        series_reduced = extract_series(metrics_reduced or [])
        summary_table = pd.DataFrame(
            [
                {
                    "Modo": "Completo",
                    **{
                        row["Campo"]: row["Valor"]
                        for row in self._build_summary_rows(summary_full, self._resumo_gap(series_full, summary_full))
                    },
                },
                {
                    "Modo": "Reduzido",
                    **{
                        row["Campo"]: row["Valor"]
                        for row in self._build_summary_rows(summary_reduced, self._resumo_gap(series_reduced, summary_reduced))
                    },
                },
            ]
        )
        st.table(summary_table)

        if not metrics_full and not metrics_reduced:
            st.warning("Nenhum arquivo .jsonl selecionado para métricas de comparação.")
            return
        if not metrics_full or not metrics_reduced:
            st.warning("Selecione arquivos .jsonl para completo e reduzido para comparar curvas.")
            return

        st.subheader("Resumo do Gap")
        resumo_full = self._resumo_gap(series_full, summary_full)
        resumo_reduced = self._resumo_gap(series_reduced, summary_reduced)
        speedup = None
        delta_quality = None
        if resumo_full.get("runtime_total_s") and resumo_reduced.get("runtime_total_s"):
            try:
                speedup = float(resumo_full["runtime_total_s"]) / float(resumo_reduced["runtime_total_s"])
            except (TypeError, ValueError, ZeroDivisionError):
                speedup = None
        if resumo_full.get("best_cheap") is not None and resumo_reduced.get("best_cheap") is not None:
            try:
                delta_quality = float(resumo_reduced["best_cheap"]) - float(resumo_full["best_cheap"])
            except (TypeError, ValueError):
                delta_quality = None
        resumo_compare = {
            "speedup": speedup,
            "delta_quality": delta_quality,
        }
        st.table(pd.DataFrame([{"Modo": "Completo", **resumo_full}, {"Modo": "Reduzido", **resumo_reduced}]))
        st.table(pd.DataFrame([resumo_compare]))

        self._render_warning_missing(series_full)
        self._render_warning_missing(series_reduced)

        st.subheader("Gráficos (Full vs Reduced)")
        temp_dir = Path(tempfile.mkdtemp(prefix="reports_compare_"))

        omega_full = temp_dir / "omega_full.png"
        omega_reduced = temp_dir / "omega_reduced.png"
        plot_omega_reduction(series_full, omega_full)
        plot_omega_reduction(series_reduced, omega_reduced)
        col1, col2 = st.columns(2)
        with col1:
            self._render_png(omega_full, "Redução Ω (Completo)")
        with col2:
            self._render_png(omega_reduced, "Redução Ω (Reduzido)")

        cost_png = temp_dir / "cost_quality_compare.png"
        plot_cost_quality({"full": series_full, "reduced": series_reduced}, cost_png)
        self._render_png(cost_png, "Custo x Qualidade (Comparativo)")

        stability_full = temp_dir / "stability_full.png"
        stability_reduced = temp_dir / "stability_reduced.png"
        plot_score_stability(series_full, stability_full)
        plot_score_stability(series_reduced, stability_reduced)
        col3, col4 = st.columns(2)
        with col3:
            self._render_png(stability_full, "Estabilidade (Completo)")
        with col4:
            self._render_png(stability_reduced, "Estabilidade (Reduzido)")

        pocket_full = temp_dir / "pocket_full.png"
        pocket_reduced = temp_dir / "pocket_reduced.png"
        plot_pocket_rank_effect(metrics_full or [], pocket_full)
        plot_pocket_rank_effect(metrics_reduced or [], pocket_reduced)
        col5, col6 = st.columns(2)
        with col5:
            self._render_png(pocket_full, "Pocket ranking (Completo)")
        with col6:
            self._render_png(pocket_reduced, "Pocket ranking (Reduzido)")

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
        base_dir: Path | None = None

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
            base_dir = selected_folder
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
            json_payloads = {name: load_any_json(path) for name, path in json_map.items()}
            main_json = json_payloads[selected_name]
            aux_jsons = {name: payload for name, payload in json_payloads.items() if name != selected_name}

            jsonl_paths = self._buscar_jsonl_na_pasta(selected_folder)
            jsonl_map = {path.name: path for path in jsonl_paths}

            kind = infer_json_kind(main_json)
            if kind == "single" and jsonl_map:
                match = find_matching_jsonl(main_json_path)
                if match and match.exists():
                    metrics_path = match
                    metrics_records = load_jsonl(metrics_path)
                else:
                    options = ["Nenhum"] + list(jsonl_map.keys())
                    selected_metrics = st.selectbox("Arquivo de métricas (.jsonl)", options)
                    if selected_metrics != "Nenhum":
                        metrics_path = jsonl_map[selected_metrics]
                        metrics_records = load_jsonl(metrics_path)
            elif kind == "compare" and jsonl_map:
                full_block = self._extract_compare_block(main_json, "full") or {}
                reduced_block = self._extract_compare_block(main_json, "reduced") or {}
                full_base = self._resolve_report_path(base_dir, full_block.get("outdir"))
                reduced_base = self._resolve_report_path(base_dir, reduced_block.get("outdir"))
                report_full_path = self._resolve_report_path(full_base, full_block.get("metrics_path"))
                report_reduced_path = self._resolve_report_path(reduced_base, reduced_block.get("metrics_path"))
                if report_full_path and report_full_path.exists():
                    metrics_full_path = report_full_path
                    metrics_full = load_jsonl(metrics_full_path)
                if report_reduced_path and report_reduced_path.exists():
                    metrics_reduced_path = report_reduced_path
                    metrics_reduced = load_jsonl(metrics_reduced_path)

                if metrics_full is None or metrics_reduced is None:
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

        kind = infer_json_kind(main_json)
        summary_data = main_json
        summary_full = main_json
        summary_reduced = main_json

        if kind == "single":
            summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, main_json.get("outdir")),
                main_json.get("summary_path"),
            )
            summary_data = self._load_summary_data(summary_path, main_json)
            if metrics_records is None:
                report_metrics_path = find_matching_jsonl(main_json_path) if main_json_path else None
                if report_metrics_path and report_metrics_path.exists():
                    metrics_path = report_metrics_path
                    metrics_records = load_jsonl(metrics_path)
        elif kind == "compare":
            full_block = self._extract_compare_block(main_json, "full") or {}
            reduced_block = self._extract_compare_block(main_json, "reduced") or {}
            full_summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, full_block.get("outdir")),
                full_block.get("summary_path"),
            )
            reduced_summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, reduced_block.get("outdir")),
                reduced_block.get("summary_path"),
            )
            summary_full = self._load_summary_data(full_summary_path, full_block or main_json)
            summary_reduced = self._load_summary_data(reduced_summary_path, reduced_block or main_json)
            if metrics_full is None:
                report_full_path = self._resolve_report_path(
                    self._resolve_report_path(base_dir, full_block.get("outdir")),
                    full_block.get("metrics_path"),
                )
                if report_full_path and report_full_path.exists():
                    metrics_full_path = report_full_path
                    metrics_full = load_jsonl(metrics_full_path)
            if metrics_reduced is None:
                report_reduced_path = self._resolve_report_path(
                    self._resolve_report_path(base_dir, reduced_block.get("outdir")),
                    reduced_block.get("metrics_path"),
                )
                if report_reduced_path and report_reduced_path.exists():
                    metrics_reduced_path = report_reduced_path
                    metrics_reduced = load_jsonl(metrics_reduced_path)

        bundle = ReportBundle(
            kind=kind,
            main_json=main_json,
            metrics=metrics_records,
            aux_jsons=aux_jsons,
        )

        if bundle.kind == "single":
            self._render_single_report(bundle, metrics_records, summary_data)
        elif bundle.kind == "compare":
            self._render_compare_report(bundle, metrics_full, metrics_reduced, summary_full, summary_reduced)
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
