from dockingpp.gui.pages.reports import ReportsPage


def test_reports_page_import_smoke() -> None:
    assert ReportsPage.id == "Relatórios"
