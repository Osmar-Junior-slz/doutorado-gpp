"""Base classes for Streamlit pages."""

from __future__ import annotations

from typing import Protocol

from dockingpp.gui.state import AppState


class BasePage(Protocol):
    """Contract for GUI pages."""

    id: str
    title: str

    def render(self, state: AppState) -> None:
        """Render the page UI."""

