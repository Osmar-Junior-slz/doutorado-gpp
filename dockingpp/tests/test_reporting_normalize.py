from dockingpp.reporting.models import ReportBundle, RunSummary
from dockingpp.reporting.normalize import optional_warnings


def test_warnings_campos_opcionais() -> None:
    summary = RunSummary(
        run_id="r",
        status="success",
        mode="single",
        engine="x",
        receptor="r",
        peptide="p",
        runtime_sec=1.0,
        omega_full=1,
        omega_reduced=1,
        omega_ratio=1,
        n_pockets_total=1,
        n_pockets_selected=1,
        n_evals_cheap=1,
        n_evals_expensive=0,
        best_score_cheap=1,
        best_score_expensive=None,
        confidence_final=None,
        trigger_count_expensive=0,
    )
    warnings = optional_warnings(ReportBundle(summary=summary))
    assert "score caro não executado" in warnings
    assert "calibração ausente" in warnings
