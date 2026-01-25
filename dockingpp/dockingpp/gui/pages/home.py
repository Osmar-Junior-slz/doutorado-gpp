"""Home page for Streamlit GUI."""

from __future__ import annotations

import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.state import AppState, StateKeys, set_state


class HomePage(BasePage):
    id = "Início"
    title = "Início"

    def render(self, state: AppState) -> None:
        st.header("Docking Reduce")
        st.write("Interface local para executar experimentos de docking e comparar busca completa vs reduzida.")
        if st.button("Novo experimento"):
            set_state(**{StateKeys.PAGE: "Docking"})
            st.rerun()

