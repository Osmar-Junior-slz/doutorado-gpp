"""Streamlit GUI entrypoint for dockingpp."""

from __future__ import annotations

import streamlit as st

from dockingpp.gui.pages import ConfigPage, DockingPage, HomePage, PdbPrepPage, ReportsPage
from dockingpp.gui.state import AppState, StateKeys, get_state, init_state_defaults, set_state


def build_pages() -> dict[str, object]:
    return {
        HomePage.id: HomePage(),
        DockingPage.id: DockingPage(),
        PdbPrepPage.id: PdbPrepPage(),
        ConfigPage.id: ConfigPage(),
        ReportsPage.id: ReportsPage(),
    }


def render_sidebar(state: AppState, pages: dict[str, object]) -> str:
    st.sidebar.title("Docking Reduce")
    page_ids = list(pages.keys())
    current_page = state.page if state.page in page_ids else HomePage.id
    selection = st.sidebar.radio("Menu", options=page_ids, index=page_ids.index(current_page))
    if selection != state.page:
        set_state(**{StateKeys.PAGE: selection})
    return selection


def main() -> None:
    """Main entrypoint for Streamlit."""

    st.set_page_config(page_title="Docking Reduce", layout="centered")
    init_state_defaults()

    pages = build_pages()
    state = get_state()
    selection = render_sidebar(state, pages)

    page = pages.get(selection)
    if page is None:
        st.error("Página não encontrada.")
        return
    page.render(get_state())


if __name__ == "__main__":
    main()

