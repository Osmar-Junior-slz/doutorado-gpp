"""Streamlit GUI entrypoint for dockingpp."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# PT-BR: streamlit run dockingpp/gui/app.py executa com sys.path relativo ao
# diretório do script; garantimos o root do projeto para imports absolutos.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = next(
    (parent for parent in _THIS_FILE.parents if (parent / "pyproject.toml").exists()),
    _THIS_FILE.parents[2],
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

