"""Página de relatórios baseada no pacote dockingpp.reporting."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.state import AppState
from dockingpp.reporting import aggregates, loaders, normalize
from dockingpp.reporting.plots import (
    plot_confidence,
    plot_convergence,
    plot_cost_vs_quality,
    plot_pockets_total_vs_selected,
    plot_search_reduction,
    plot_trigger_timeline,
)


class ReportsPage(BasePage):
    id = "Relatórios"
    title = "Relatórios"

    def render(self, state: AppState) -> None:
        st.title(self.title)
        base = st.text_input("Diretório com saídas", value="out")
        run_dirs = loaders.discover_run_dirs(Path(base))
        if not run_dirs:
            st.info("Nenhuma execução encontrada (summary.json).")
            return

        run_dir = st.selectbox("Execução", options=run_dirs, format_func=lambda p: p.name)
        bundle = loaders.load_report_bundle(Path(run_dir))

        st.subheader("Resumo")
        st.json(bundle.summary.model_dump(mode="json"))

        for warning in normalize.optional_warnings(bundle):
            st.warning(warning)

        series = normalize.to_timeseries(bundle)
        reduction = aggregates.compute_search_reduction(bundle)
        cost_quality = aggregates.compute_cost_quality(bundle)
        pockets = aggregates.compute_pockets(bundle)

        st.pyplot(plot_search_reduction(**reduction))
        st.pyplot(plot_convergence(series.get("iter", []), series.get("best_cheap", [])))
        st.pyplot(plot_cost_vs_quality(**cost_quality))
        st.pyplot(plot_trigger_timeline(series.get("trigger_timeline", [])))
        st.pyplot(plot_pockets_total_vs_selected(**pockets))
        st.pyplot(plot_confidence(bundle.summary.confidence_final))
