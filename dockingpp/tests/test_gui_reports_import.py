"""Smoke test para importação da página de relatórios (PT-BR)."""

from __future__ import annotations


def test_import_reports_page() -> None:
    """Importa reports.py sem erro."""

    from dockingpp.gui.pages import reports  # noqa: F401

    assert reports.ReportsPage.id == "Relatórios"
