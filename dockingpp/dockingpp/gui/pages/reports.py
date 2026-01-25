"""Reports page."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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

    @staticmethod
    def _is_monotonic(values: list[float]) -> bool:
        return all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))

    @staticmethod
    def _series_values(series: list[dict[str, float]]) -> list[float]:
        return [float(item["score"]) for item in series if item.get("score") is not None]

    @staticmethod
    def _prepare_series(series: list[dict[str, float]]) -> list[dict[str, float]]:
        return sorted(series, key=lambda item: item.get("step", 0))

    @staticmethod
    def _compute_evals_cumulative(series: list[dict[str, float]]) -> list[dict[str, float]]:
        ordered = ReportsPage._prepare_series(series)
        values = ReportsPage._series_values(ordered)
        if not values:
            return []
        if ReportsPage._is_monotonic(values):
            return [{"step": item["step"], "score": float(item["score"])} for item in ordered]
        cumulative = []
        total = 0.0
        for item in ordered:
            value = float(item["score"])
            total += value
            cumulative.append({"step": item["step"], "score": total})
        return cumulative

    @staticmethod
    def _pair_best_vs_evals(
        best_series: list[dict[str, float]],
        evals_cumulative: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        best_by_step = {
            item["step"]: float(item["score"]) for item in best_series if item.get("score") is not None
        }
        paired = []
        for item in evals_cumulative:
            step = item.get("step")
            if step in best_by_step and item.get("score") is not None:
                paired.append({"step": float(item["score"]), "score": best_by_step[step]})
        return paired

    @staticmethod
    def _compute_total_evals(series: list[dict[str, float]]) -> float | None:
        values = ReportsPage._series_values(series)
        if not values:
            return None
        if ReportsPage._is_monotonic(values):
            return values[-1]
        return float(sum(values))

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
    def _plot_bar(values: dict[str, float], title: str, ylabel: str) -> plt.Figure:
        fig, ax = plt.subplots()
        labels = list(values.keys())
        heights = list(values.values())
        ax.bar(labels, heights)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return fig

    @staticmethod
    def _save_figure(fig: plt.Figure, save_dir: Path, filename: str) -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        return path

    @staticmethod
    def _download_csv_button(label: str, df: pd.DataFrame, filename: str) -> None:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(label, csv_data, file_name=filename, mime="text/csv")

    @staticmethod
    def _resolve_compare_dir(report_data: dict[str, object] | None, base_dir: Path, label: str) -> Path | None:
        if report_data and label in report_data:
            run_data = report_data[label]
            if isinstance(run_data, dict):
                for key in ("out_dir", "output_dir", "path", "dir"):
                    value = run_data.get(key)
                    if value:
                        resolved = Path(value)
                        if not resolved.is_absolute():
                            resolved = base_dir / resolved
                        if resolved.exists():
                            return resolved
        candidate = base_dir / label
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def _load_run_payload(run_dir: Path) -> dict[str, object]:
        result_path = run_dir / "result.json"
        metrics_path = run_dir / "metrics.jsonl"
        metrics_records = load_jsonl(metrics_path)
        metrics_summary = summarize_metrics(metrics_records)
        result_data = load_json(result_path) if result_path.exists() else {}
        eval_series, _ = metrics_series(metrics_records, ["n_eval", "evals", "evaluations"])
        total_evals = ReportsPage._compute_total_evals(eval_series)
        best_score = result_data.get("best_score_cheap") or result_data.get("best_score")
        elapsed = result_data.get("elapsed_s") or result_data.get("elapsed_seconds")
        return {
            "result": result_data,
            "metrics_records": metrics_records,
            "metrics_summary": metrics_summary,
            "best_score": best_score,
            "elapsed": elapsed,
            "total_evals": total_evals,
        }

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

        st.subheader("Seleção de execuções")
        selected = st.selectbox("Execução", options=options, index=default_index)
        selected_path = runs[options.index(selected)]
        multi_selected = st.multiselect("Selecionar múltiplas execuções para análise em lote", options=options)
        multi_paths = [runs[options.index(option)] for option in multi_selected]

        report_path = selected_path / "report.json"
        result_path = selected_path / "result.json"
        metrics_path = selected_path / "metrics.jsonl"

        report_data = load_json(report_path) if report_path.exists() else None
        full_dir = self._resolve_compare_dir(report_data, selected_path, "full")
        reduced_dir = self._resolve_compare_dir(report_data, selected_path, "reduced")
        compare_mode = report_data is not None or (full_dir is not None and reduced_dir is not None)

        metrics_records = load_jsonl(metrics_path)
        metrics_summary = summarize_metrics(metrics_records)
        result_data = load_json(result_path) if result_path.exists() else {}

        if not metrics_path.exists() and not compare_mode:
            st.warning("metrics.jsonl não encontrado nesta execução.")

        if compare_mode:
            st.subheader("Comparação: Completo vs Reduzido")
            rows = build_compare_table(report_data) if report_data else []
            if not rows and full_dir and reduced_dir:
                full_payload = self._load_run_payload(full_dir)
                reduced_payload = self._load_run_payload(reduced_dir)
                rows = [
                    {
                        "Modo": "Completo",
                        "Melhor score (cheap)": full_payload["best_score"],
                        "Avaliações": full_payload["total_evals"],
                        "Bolsões totais": full_payload["metrics_summary"].get("n_pockets_total"),
                        "Bolsões usados": full_payload["metrics_summary"].get("n_pockets_used"),
                        "Razão de redução": full_payload["metrics_summary"].get("reduction_ratio"),
                        "Tempo (s)": full_payload["elapsed"],
                    },
                    {
                        "Modo": "Reduzido",
                        "Melhor score (cheap)": reduced_payload["best_score"],
                        "Avaliações": reduced_payload["total_evals"],
                        "Bolsões totais": reduced_payload["metrics_summary"].get("n_pockets_total"),
                        "Bolsões usados": reduced_payload["metrics_summary"].get("n_pockets_used"),
                        "Razão de redução": reduced_payload["metrics_summary"].get("reduction_ratio"),
                        "Tempo (s)": reduced_payload["elapsed"],
                    },
                ]
            if rows:
                st.table(rows)
                compare_df = pd.DataFrame(rows)
                self._download_csv_button("Baixar tabela resumo (CSV)", compare_df, "comparacao_full_reduced.csv")
            else:
                st.warning("report.json encontrado, mas sem dados de comparação completos.")

            if full_dir and reduced_dir:
                full_payload = self._load_run_payload(full_dir)
                reduced_payload = self._load_run_payload(reduced_dir)
                metrics_full = full_payload["metrics_records"]
                metrics_reduced = reduced_payload["metrics_records"]

                eval_full = full_payload["total_evals"] or full_payload["metrics_summary"].get("n_eval")
                eval_reduced = reduced_payload["total_evals"] or reduced_payload["metrics_summary"].get("n_eval")
                elapsed_full = (
                    report_data.get("full", {}).get("elapsed_seconds") if report_data else full_payload["elapsed"]
                )
                elapsed_reduced = (
                    report_data.get("reduced", {}).get("elapsed_seconds") if report_data else reduced_payload["elapsed"]
                )

                st.subheader("Indicadores comparativos")
                charts_to_save: list[tuple[str, plt.Figure]] = []
                eval_chart = self._plot_bar(
                    {"Completo": float(eval_full) if eval_full is not None else 0.0, "Reduzido": float(eval_reduced) if eval_reduced is not None else 0.0},
                    "Avaliações",
                    "Avaliações",
                )
                st.pyplot(eval_chart)
                charts_to_save.append(("avaliacoes_full_reduced.png", eval_chart))

                if elapsed_full is not None or elapsed_reduced is not None:
                    time_chart = self._plot_bar(
                        {
                            "Completo": float(elapsed_full) if elapsed_full is not None else 0.0,
                            "Reduzido": float(elapsed_reduced) if elapsed_reduced is not None else 0.0,
                        },
                        "Tempo (s)",
                        "Tempo (s)",
                    )
                    st.pyplot(time_chart)
                    charts_to_save.append(("tempo_full_reduced.png", time_chart))

                pockets_chart = self._plot_bar(
                    {
                        "Completo": float(full_payload["metrics_summary"].get("n_pockets_used") or 0.0),
                        "Reduzido": float(reduced_payload["metrics_summary"].get("n_pockets_used") or 0.0),
                    },
                    "Bolsões usados",
                    "Bolsões",
                )
                st.pyplot(pockets_chart)
                charts_to_save.append(("bolsoes_full_reduced.png", pockets_chart))

                ratio_chart = self._plot_bar(
                    {
                        "Completo": float(full_payload["metrics_summary"].get("reduction_ratio") or 0.0),
                        "Reduzido": float(reduced_payload["metrics_summary"].get("reduction_ratio") or 0.0),
                    },
                    "Razão de redução",
                    "Razão",
                )
                st.pyplot(ratio_chart)
                charts_to_save.append(("reducao_full_reduced.png", ratio_chart))

                st.subheader("Convergência (Full vs Reduced)")
                full_best, _ = metrics_series(metrics_full, ["best_score_cheap", "best_score", "best"])
                reduced_best, _ = metrics_series(metrics_reduced, ["best_score_cheap", "best_score", "best"])
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
                    charts_to_save.append(("convergencia_full_reduced.png", fig))
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
                    charts_to_save.append(("convergencia_avaliacoes_full_reduced.png", fig))
                else:
                    st.warning("n_eval ou best score não disponíveis para comparação por avaliações.")

                st.subheader("Trade-off")
                trade_x = []
                trade_y = []
                labels = []
                if eval_full is not None and eval_reduced is not None:
                    trade_x = [float(eval_full), float(eval_reduced)]
                    trade_y = [
                        float(full_payload["best_score"]) if full_payload["best_score"] is not None else 0.0,
                        float(reduced_payload["best_score"]) if reduced_payload["best_score"] is not None else 0.0,
                    ]
                    labels = ["Completo", "Reduzido"]
                    fig = self._plot_scatter(
                        trade_x,
                        trade_y,
                        labels,
                        "Trade-off (avaliações)",
                        "Avaliações",
                        "Best score",
                    )
                    st.pyplot(fig)
                    charts_to_save.append(("tradeoff_full_reduced.png", fig))
                else:
                    st.warning("Dados insuficientes para o scatter de trade-off.")

                if st.button("Salvar gráficos (PNG)"):
                    save_dir = selected_path / "figures"
                    for filename, fig in charts_to_save:
                        path = self._save_figure(fig, save_dir, filename)
                        with open(path, "rb") as handle:
                            st.download_button(
                                f"Baixar {filename}",
                                handle.read(),
                                file_name=filename,
                                mime="image/png",
                            )
                        st.caption(f"Figura salva em: {path}")
                for _, fig in charts_to_save:
                    plt.close(fig)
        else:
            st.warning(
                "Execução única: report.json não é gerado. Para gerar, rode no modo "
                "'Comparar (Completo vs Reduzido)'."
            )

        st.subheader("Resumo da execução")
        summary_rows = [
            {"Campo": "Melhor score (cheap)", "Valor": result_data.get("best_score_cheap") or result_data.get("best_score")},
            {"Campo": "Avaliações", "Valor": self._compute_total_evals(metrics_series(metrics_records, ["n_eval", "evals", "evaluations"])[0])},
            {"Campo": "Bolsões usados", "Valor": metrics_summary.get("n_pockets_used")},
            {"Campo": "Razão de redução", "Valor": metrics_summary.get("reduction_ratio")},
            {"Campo": "Tempo (s)", "Valor": result_data.get("elapsed_s") or result_data.get("elapsed_seconds")},
        ]
        st.table(summary_rows)
        summary_df = pd.DataFrame(summary_rows)
        self._download_csv_button("Baixar tabela resumo (CSV)", summary_df, "resumo_execucao.csv")

        st.subheader("Convergência do score")
        best_series, _ = metrics_series(metrics_records, ["best_score_cheap", "best_score", "best"])
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
            st.warning("Não foi possível encontrar best score em metrics.jsonl.")

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

        st.subheader("Estatísticas por geração")
        mean_series, _ = metrics_series(metrics_records, ["mean_score_cheap", "mean_score"])
        if mean_series:
            mean_series = self._prepare_series(mean_series)
            fig = self._plot_line(
                mean_series,
                "step",
                "score",
                "Mean score vs geração",
                "Geração",
                "Mean score",
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("mean_score não disponível para plotar.")

        diversity_series, _ = metrics_series(metrics_records, ["diversity", "diversidade"])
        if diversity_series:
            diversity_series = self._prepare_series(diversity_series)
            fig = self._plot_line(
                diversity_series,
                "step",
                "score",
                "Diversidade vs geração",
                "Geração",
                "Diversidade",
            )
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Custo × Qualidade")
        total_evals = self._compute_total_evals(eval_series)
        total_time = result_data.get("elapsed_s") or result_data.get("elapsed_seconds")
        trade_x = total_time if total_time is not None else total_evals
        trade_label = "Tempo (s)" if total_time is not None else "Avaliações acumuladas"
        if trade_x is not None and best_series:
            last_best = best_series[-1]["score"]
            fig = self._plot_scatter(
                [float(trade_x)],
                [float(last_best)],
                ["Execução única"],
                "Trade-off custo × qualidade",
                trade_label,
                "Best score",
            )
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Ponto único: selecione múltiplas execuções para comparar trade-offs.")
        else:
            st.warning("Dados insuficientes para o gráfico de trade-off.")

        if multi_paths:
            st.subheader("Análise em lote")
            rows = []
            for run_path in multi_paths:
                payload = self._load_run_payload(run_path)
                rows.append(
                    {
                        "Execução": run_path.name,
                        "Best score": payload["best_score"],
                        "Avaliações": payload["total_evals"],
                        "Tempo (s)": payload["elapsed"],
                        "Razão de redução": payload["metrics_summary"].get("reduction_ratio"),
                    }
                )
            multi_df = pd.DataFrame(rows)
            for col in ["Best score", "Avaliações", "Tempo (s)", "Razão de redução"]:
                multi_df[col] = pd.to_numeric(multi_df[col], errors="coerce")
            st.table(multi_df)
            self._download_csv_button("Baixar tabela resumo (CSV)", multi_df, "resumo_multiplas_execucoes.csv")

            stats_df = multi_df.drop(columns=["Execução"]).agg(["mean", "median", "std"])
            st.write("Estatísticas agregadas")
            st.table(stats_df)

            if len(multi_df) > 1:
                fig, ax = plt.subplots()
                ax.boxplot(multi_df["Best score"].dropna())
                ax.set_title("Distribuição do best score")
                ax.set_ylabel("Best score")
                st.pyplot(fig)
                plt.close(fig)

            if multi_df["Razão de redução"].notna().any() and multi_df["Best score"].notna().any():
                fig = self._plot_scatter(
                    multi_df["Razão de redução"].fillna(0.0).tolist(),
                    multi_df["Best score"].fillna(0.0).tolist(),
                    multi_df["Execução"].tolist(),
                    "Trade-off em lote (redução vs best score)",
                    "Razão de redução",
                    "Best score",
                )
                st.pyplot(fig)
                plt.close(fig)

            if multi_df["Avaliações"].notna().any() and multi_df["Best score"].notna().any():
                fig = self._plot_scatter(
                    multi_df["Avaliações"].fillna(0.0).tolist(),
                    multi_df["Best score"].fillna(0.0).tolist(),
                    multi_df["Execução"].tolist(),
                    "Trade-off em lote (avaliações vs best score)",
                    "Avaliações",
                    "Best score",
                )
                st.pyplot(fig)
                plt.close(fig)

        st.write("Downloads")
        download_json_button("Baixar result.json", result_path, filename="result.json", warn_missing=False)
        download_json_button("Baixar report.json", report_path, filename="report.json", warn_missing=False)
        download_json_button("Baixar metrics.jsonl", metrics_path, filename="metrics.jsonl", warn_missing=False)
