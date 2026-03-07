from dockingpp.reporting.aggregates import compute_cost_quality, compute_pockets, compute_search_reduction
from dockingpp.reporting.models import ReportBundle, RunSummary


def _bundle() -> ReportBundle:
    s = RunSummary(
        run_id="r",
        status="success",
        mode="single",
        engine="x",
        receptor="r",
        peptide="p",
        runtime_sec=1,
        omega_full=10,
        omega_reduced=2,
        omega_ratio=0.2,
        n_pockets_total=10,
        n_pockets_selected=2,
        n_evals_cheap=50,
        n_evals_expensive=5,
        best_score_cheap=1.0,
        best_score_expensive=1.2,
        confidence_final=0.9,
        trigger_count_expensive=5,
    )
    return ReportBundle(summary=s)


def test_aggregates_basicos() -> None:
    b = _bundle()
    assert compute_search_reduction(b)["omega_ratio"] == 0.2
    assert compute_pockets(b)["n_pockets_selected"] == 2
    assert compute_cost_quality(b)["n_evals_expensive"] == 5
