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
from dockingpp.reporting.loaders import (
    ReportRun,
    extract_series,
    find_matching_jsonl,
    find_report_runs,
    load_any_json,
    load_jsonl,
    pair_full_reduced,
)
from dockingpp.reporting.plots import (
    plot_convergence,
    plot_cost_comparison,
    plot_paired_comparison,
    plot_pocket_rank_effect,
    plot_search_space_reduction,
)


class ReportsPage(BasePage):
    """Página de relatórios com gráficos de convergência e comparações."""

    id = "Relatórios"
    title = "Relatórios"

    @staticmethod
    def _download_csv_button(label: str, df: pd.DataFrame, filename: str, key_suffix: str) -> None:
        """Gera um botão de download para CSV a partir de um DataFrame."""

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label,
            csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download_csv_{key_suffix}",
        )

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

    @staticmethod
    def _merge_summary_result(summary: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Mescla dados de summary e result garantindo chaves úteis no topo."""

        merged: dict[str, Any] = {}
        merged.update(result or {})
        merged.update(summary or {})
        timing = summary.get("timing") if isinstance(summary.get("timing"), dict) else {}
        if not timing and isinstance(result.get("timing"), dict):
            timing = result.get("timing")
        if isinstance(timing, dict):
            merged.setdefault("total_s", timing.get("total_s"))
            merged.setdefault("elapsed_s", timing.get("elapsed_s"))
        return merged

    @staticmethod
    def _has_time_series(records: list[dict[str, Any]] | None) -> bool:
        """Indica se há série temporal real (mais de um ponto)."""

        if not records:
            return False
        if len(records) <= 1:
            return False
        series = extract_series(records)
        return len(set(series.get("iter", []))) > 1

    @staticmethod
    def _select_series_records(
        metrics_records: list[dict[str, Any]] | None,
        metrics_timeseries: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Seleciona o arquivo com série temporal preferencial."""

        return metrics_timeseries or metrics_records

    @staticmethod
    def _is_reduced_aggregate_fallback(summary: dict[str, Any]) -> bool:
        return (
            summary.get("mode") == "reduced_aggregate"
            and bool(summary.get("fallback_to_full"))
            and not bool(summary.get("per_pocket_results") or [])
        )

    @staticmethod
    def _resolve_compare_fallback_metrics(
        summary: dict[str, Any],
        compare_block: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(summary)
        if not compare_block:
            return resolved
        for dst, keys in {
            "n_eval_total": ["reduced_total_n_eval", "total_n_eval", "n_eval_total"],
            "runtime_total_s": ["reduced_total_runtime_sec", "total_runtime_sec", "runtime_total_s"],
            "best_cheap": ["reduced_best_over_pockets_cheap", "best_over_pockets_cheap", "best_score_cheap"],
        }.items():
            for key in keys:
                if compare_block.get(key) is not None:
                    resolved[dst] = compare_block.get(key)
                    break
        return resolved


    @staticmethod
    def _load_paired_comparison_csv(csv_path: Path | None) -> list[dict[str, Any]]:
        """Carrega paired_comparison.csv quando disponível."""

        if csv_path is None or not csv_path.exists():
            return []
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.EmptyDataError):
            return []
        return frame.to_dict(orient="records")

    def _load_fallback_metrics_series(
        self,
        base_dir: Path | None,
        reduced_summary: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """Carrega métricas do fallback_full quando reduced agregou via fallback."""

        if not bool(reduced_summary.get("fallback_to_full")):
            return None, None
        fallback_outdir = reduced_summary.get("fallback_full_outdir")
        if not fallback_outdir:
            return None, None
        fallback_dir = self._resolve_report_path(base_dir, str(fallback_outdir))
        if fallback_dir is None or not fallback_dir.exists():
            return None, None
        metrics_ts_path = fallback_dir / "metrics.timeseries.jsonl"
        metrics_path = fallback_dir / "metrics.jsonl"
        ts_records = load_jsonl(metrics_ts_path) if metrics_ts_path.exists() else None
        records = load_jsonl(metrics_path) if metrics_path.exists() else None
        return records, ts_records


    def _render_kpis(self, summary_data: dict[str, Any], series: dict[str, Any] | None, key_suffix: str) -> None:
        st.subheader("Resumo do Gap")
        resumo = self._resumo_gap(series or {}, summary_data)
        st.table(pd.DataFrame([resumo]))

        st.subheader("Resumo da execução")
        summary_rows = self._build_summary_rows(summary_data, resumo)
        st.table(summary_rows)
        summary_df = pd.DataFrame(summary_rows)
        self._download_csv_button("Baixar tabela resumo (CSV)", summary_df, "resumo_execucao.csv", key_suffix)

    def _render_single_report(
        self,
        bundle: ReportBundle,
        metrics_records: list[dict[str, Any]] | None,
        summary_data: dict[str, Any],
        metrics_timeseries: list[dict[str, Any]] | None = None,
        key_suffix: str = "single",
    ) -> None:
        """Renderiza relatório de execução única com gráficos e resumos."""

        series_records = self._select_series_records(metrics_records, metrics_timeseries)
        series = extract_series(series_records or [])
        self._render_kpis(summary_data, series, key_suffix)

        if not self._has_time_series(series_records):
            st.info("Sem série temporal; exibindo KPIs.")
            return

        self._render_warning_missing(series)
        st.subheader("Gráficos")
        temp_dir = Path(tempfile.mkdtemp(prefix="reports_"))

        reduction_png = temp_dir / "search_space_reduction.png"
        if plot_search_space_reduction({**summary_data, **self._resumo_gap(series, summary_data)}, reduction_png):
            self._render_png(reduction_png, "Redução do espaço de busca")

        convergence_png = temp_dir / "convergence.png"
        if plot_convergence(series, convergence_png):
            self._render_png(convergence_png, "Convergência: best_score_cheap vs n_eval_cumulative")


    def _render_reduced_aggregate_report(
        self,
        bundle: ReportBundle,
        metrics_records: list[dict[str, Any]] | None,
        summary_data: dict[str, Any],
    ) -> None:
        """Renderiza relatório agregado do reduced por bolsão."""

        st.subheader("Reduced agregado por bolsão")
        st.json(
            {
                "best_pocket_id": summary_data.get("best_pocket_id"),
                "best_over_pockets_cheap": summary_data.get("best_over_pockets_cheap"),
                "best_over_pockets_expensive": summary_data.get("best_over_pockets_expensive"),
                "total_n_eval": summary_data.get("total_n_eval"),
                "total_runtime_sec": summary_data.get("total_runtime_sec"),
                "budget_policy": summary_data.get("budget_policy"),
                "budget_delta": summary_data.get("budget_delta"),
                "fallback_to_full": summary_data.get("fallback_to_full"),
                "fallback_reason": summary_data.get("fallback_reason"),
                "executed_mode": summary_data.get("executed_mode"),
                "n_pockets_total": summary_data.get("n_pockets_total"),
                "n_pockets_used": summary_data.get("n_pockets_used"),
            }
        )
        if bool(summary_data.get("fallback_to_full")):
            st.warning("Reduced sem bolsões viáveis; execução caiu em fallback para full")
            st.caption(f"fallback_reason={summary_data.get('fallback_reason')} | n_pockets_used={summary_data.get('n_pockets_used')}")
        per_pocket = summary_data.get("per_pocket_results", [])
        if isinstance(per_pocket, list) and per_pocket:
            frame = pd.DataFrame(per_pocket)
            st.subheader("Resultados por bolsão")
            st.dataframe(frame)
        else:
            st.info("Sem resultados por bolsão no summary agregado.")

        rejected = summary_data.get("rejected_pockets", [])
        if isinstance(rejected, list) and rejected:
            st.subheader("Bolsões rejeitados")
            st.dataframe(pd.DataFrame(rejected))

        temp_dir = Path(tempfile.mkdtemp(prefix="reports_reduced_agg_"))
        pocket_png = temp_dir / "pocket_rank_effect.png"
        if per_pocket:
            if plot_pocket_rank_effect(summary_data, pocket_png) is None and pocket_png.exists():
                self._render_png(pocket_png, "Custo/score por bolsão")
            elif pocket_png.exists():
                self._render_png(pocket_png, "Custo/score por bolsão")

    def _render_compare_report(
        self,
        bundle: ReportBundle,
        metrics_full: list[dict[str, Any]] | None,
        metrics_reduced: list[dict[str, Any]] | None,
        summary_full: dict[str, Any],
        summary_reduced: dict[str, Any],
        metrics_full_timeseries: list[dict[str, Any]] | None = None,
        metrics_reduced_timeseries: list[dict[str, Any]] | None = None,
        key_suffix: str = "compare",
    ) -> None:
        """Renderiza relatório comparativo entre modos completo e reduzido."""

        st.subheader("Comparação: Completo vs Reduzido")
        if self._is_reduced_aggregate_fallback(summary_reduced):
            st.warning("Reduced sem bolsões viáveis; execução caiu em fallback para full")
            st.caption(
                f"fallback_reason={summary_reduced.get('fallback_reason')} | "
                f"n_pockets_used={summary_reduced.get('n_pockets_used')}"
            )
            rejected = summary_reduced.get("rejected_pockets")
            if isinstance(rejected, list) and rejected:
                st.caption(f"rejected_pockets={len(rejected)}")

        rows = build_compare_table(bundle.main_json)
        if rows:
            st.subheader("Comparação avançada")
            st.table(rows)
            compare_df = pd.DataFrame(rows)
            self._download_csv_button(
                "Baixar tabela resumo (CSV)",
                compare_df,
                "comparacao_full_reduced.csv",
                f"{key_suffix}_full_reduced",
            )

        series_full_records = self._select_series_records(metrics_full, metrics_full_timeseries)
        series_reduced_records = self._select_series_records(metrics_reduced, metrics_reduced_timeseries)
        series_full = extract_series(series_full_records or [])
        series_reduced = extract_series(series_reduced_records or [])
        resumo_full = self._resumo_gap(series_full, summary_full)
        resumo_reduced = self._resumo_gap(series_reduced, summary_reduced)
        if self._is_reduced_aggregate_fallback(summary_reduced):
            reduced_block = self._extract_compare_block(bundle.main_json, "reduced")
            resumo_reduced = self._resolve_compare_fallback_metrics(resumo_reduced, reduced_block)

        st.subheader("Tabela comparativa (KPIs)")
        st.table(pd.DataFrame([{"Modo": "Completo", **resumo_full}, {"Modo": "Reduzido", **resumo_reduced}]))

        self._render_warning_missing(series_full)
        self._render_warning_missing(series_reduced)

        st.subheader("Gráficos (Full vs Reduced)")
        temp_dir = Path(tempfile.mkdtemp(prefix="reports_compare_"))

        cost_png = temp_dir / "cost_comparison.png"
        if plot_cost_comparison(resumo_full, resumo_reduced, cost_png):
            self._render_png(cost_png, "Comparação de custo: runtime_sec e n_eval_total (Reduced pode estar em fallback→full)")
        else:
            st.info("Sem dados suficientes para gráfico de custo.")

        reduction_full_png = temp_dir / "reduction_full.png"
        reduction_reduced_png = temp_dir / "reduction_reduced.png"
        has_full_reduction = plot_search_space_reduction({**summary_full, **resumo_full}, reduction_full_png)
        if self._is_reduced_aggregate_fallback(summary_reduced):
            has_reduced_reduction = False
        else:
            has_reduced_reduction = plot_search_space_reduction({**summary_reduced, **resumo_reduced}, reduction_reduced_png)
        if has_full_reduction or has_reduced_reduction:
            col1, col2 = st.columns(2)
            with col1:
                if has_full_reduction:
                    self._render_png(reduction_full_png, "Redução do espaço de busca (Completo)")
            with col2:
                if has_reduced_reduction:
                    self._render_png(reduction_reduced_png, "Redução do espaço de busca (Reduzido)")
        else:
            st.info("Sem dados suficientes para redução do espaço de busca.")

        convergence_png = temp_dir / "convergence_compare.png"
        reduced_series_for_plot = series_reduced
        if self._is_reduced_aggregate_fallback(summary_reduced) and not metrics_reduced and not metrics_reduced_timeseries:
            st.info("Fallback no reduced sem dados de série temporal; curva Reduced ocultada.")
            reduced_series_for_plot = {}
        if plot_convergence({"full": series_full, "reduced": reduced_series_for_plot}, convergence_png):
            self._render_png(convergence_png, "Convergência: best_score_cheap vs n_eval_cumulative")
        else:
            st.info("Sem dados suficientes para convergência.")

        paired_rows: list[dict[str, Any]] = []
        paired_obj = bundle.aux_jsons.get("paired_comparison")
        if isinstance(paired_obj, dict):
            if all(k in paired_obj for k in ("speedup_runtime", "speedup_eval", "delta_score_cheap")):
                paired_rows = [paired_obj]
            elif isinstance(paired_obj.get("rows"), list):
                paired_rows = [row for row in paired_obj.get("rows", []) if isinstance(row, dict)]
        paired_png = temp_dir / "paired_comparison.png"
        if plot_paired_comparison(paired_rows, paired_png):
            self._render_png(paired_png, "Comparação pareada: speedup_runtime, speedup_eval, delta_score_cheap")
        elif paired_rows:
            st.info("Dados pareados insuficientes para plot.")

    @staticmethod
    def _load_run_payload(path: Path | None) -> dict[str, Any]:
        if path and path.exists():
            try:
                return load_any_json(path)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _render_run_block(
        self,
        label: str,
        run: ReportRun,
    ) -> None:
        st.subheader(label)
        summary_payload = self._load_run_payload(run.summary_path)
        result_payload = self._load_run_payload(run.result_path)
        merged = self._merge_summary_result(summary_payload, result_payload)
        metrics_records = load_jsonl(run.metrics_path) if run.metrics_path else None
        metrics_timeseries = load_jsonl(run.metrics_timeseries_path) if run.metrics_timeseries_path else None
        self._render_single_report(
            ReportBundle(kind="single", main_json=merged, metrics=metrics_records, aux_jsons={}),
            metrics_records,
            merged,
            metrics_timeseries,
            key_suffix=f"{run.kind}_{run.run_id or run.run_dir.name}",
        )

    def _render_folder_reports(self, selected_folder: Path) -> None:
        runs = find_report_runs(selected_folder)
        if not runs:
            st.warning("Nenhuma execução encontrada na pasta.")
            return

        view_mode = st.selectbox("Modo de visualização", ["Separado", "Comparar"])

        full_runs = [run for run in runs if run.kind == "full"]
        reduced_runs = [run for run in runs if run.kind == "reduced"]

        def _select_run(label: str, options: list[ReportRun]) -> ReportRun | None:
            if not options:
                st.warning(f"Nenhum run {label.lower()} encontrado.")
                return None
            labels = [run.label() for run in options]
            selected = st.selectbox(f"Run {label}", labels, key=f"run_{label}")
            return options[labels.index(selected)]

        if view_mode == "Separado":
            if full_runs or reduced_runs:
                selected_full = _select_run("Full", full_runs) if full_runs else None
                selected_reduced = _select_run("Reduced", reduced_runs) if reduced_runs else None
                if selected_full:
                    self._render_run_block("Full", selected_full)
                if selected_reduced:
                    self._render_run_block("Reduced", selected_reduced)
                return

            unknown_runs = [run for run in runs if run.kind == "unknown"]
            selected_unknown = _select_run("Run", unknown_runs)
            if selected_unknown:
                self._render_run_block("Run", selected_unknown)
            return

        if view_mode == "Comparar":
            selected_full, selected_reduced = pair_full_reduced(runs)
            if full_runs:
                selected_full = _select_run("Full", full_runs) or selected_full
            if reduced_runs:
                selected_reduced = _select_run("Reduced", reduced_runs) or selected_reduced
            if not selected_full or not selected_reduced:
                st.warning("É necessário selecionar runs Full e Reduced para comparar.")
                return

            summary_full = self._merge_summary_result(
                self._load_run_payload(selected_full.summary_path),
                self._load_run_payload(selected_full.result_path),
            )
            summary_reduced = self._merge_summary_result(
                self._load_run_payload(selected_reduced.summary_path),
                self._load_run_payload(selected_reduced.result_path),
            )
            metrics_full = load_jsonl(selected_full.metrics_path) if selected_full.metrics_path else None
            metrics_reduced = load_jsonl(selected_reduced.metrics_path) if selected_reduced.metrics_path else None
            metrics_full_timeseries = (
                load_jsonl(selected_full.metrics_timeseries_path) if selected_full.metrics_timeseries_path else None
            )
            metrics_reduced_timeseries = (
                load_jsonl(selected_reduced.metrics_timeseries_path) if selected_reduced.metrics_timeseries_path else None
            )
            paired_rows: list[dict[str, Any]] = []
            for candidate in [
                selected_full.run_dir / "paired_comparison.csv",
                selected_reduced.run_dir / "paired_comparison.csv",
                selected_full.run_dir.parent / "paired_comparison.csv",
                selected_folder / "paired_comparison.csv",
            ]:
                paired_rows = self._load_paired_comparison_csv(candidate)
                if paired_rows:
                    break

            self._render_compare_report(
                ReportBundle(kind="compare", main_json={}, metrics=None, aux_jsons={"paired_comparison": {"rows": paired_rows}}),
                metrics_full,
                metrics_reduced,
                summary_full,
                summary_reduced,
                metrics_full_timeseries,
                metrics_reduced_timeseries,
            )
            return

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
            self._render_folder_reports(selected_folder)
            return
        else:
            main_json_upload = st.file_uploader("JSON principal", type=["json"])
            if not main_json_upload:
                st.info("Selecione um arquivo JSON para continuar.")
                return
            main_json = self._load_uploaded_json(main_json_upload)
            kind = infer_json_kind(main_json)

            if kind in {"single", "reduced_aggregate"}:
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

        if kind in {"single", "reduced_aggregate"}:
            summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, main_json.get("outdir")),
                main_json.get("summary_path"),
            )
            summary_data = self._load_summary_data(summary_path, main_json)
            if metrics_records is None:
                report_metrics_path = self._resolve_report_path(
                    self._resolve_report_path(base_dir, summary_data.get("outdir")),
                    summary_data.get("metrics_path"),
                )
                if report_metrics_path and report_metrics_path.exists():
                    metrics_path = report_metrics_path
                    metrics_records = load_jsonl(metrics_path)
                elif main_json_path:
                    report_metrics_path = find_matching_jsonl(main_json_path)
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
                else:
                    fallback_records, fallback_timeseries = self._load_fallback_metrics_series(base_dir, summary_reduced)
                    if fallback_records is not None:
                        metrics_reduced = fallback_records
                    if fallback_timeseries is not None:
                        metrics_reduced = fallback_timeseries

        bundle = ReportBundle(
            kind=kind,
            main_json=main_json,
            metrics=metrics_records,
            aux_jsons=aux_jsons,
        )

        if bundle.kind == "single":
            self._render_single_report(bundle, metrics_records, summary_data)
        elif bundle.kind == "reduced_aggregate":
            self._render_reduced_aggregate_report(bundle, metrics_records, summary_data)
        elif bundle.kind == "compare":
            self._render_compare_report(
                bundle,
                metrics_full,
                metrics_reduced,
                summary_full,
                summary_reduced,
            )
        else:
            st.warning("Não foi possível identificar o tipo do JSON. Exibindo conteúdo bruto.")
            st.json(bundle.main_json)

        st.subheader("Downloads")
        if main_json_path:
            download_json_button("Baixar JSON principal", main_json_path, filename=main_json_path.name, warn_missing=False)
        elif main_json_upload:
            self._download_json_payload("Baixar JSON principal", main_json, main_json_upload.name)

        if bundle.kind in {"single", "reduced_aggregate"}:
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
