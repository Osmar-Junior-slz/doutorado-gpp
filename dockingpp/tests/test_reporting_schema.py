from dockingpp.reporting.schema import build_bundle


def test_summary_valido_schema() -> None:
    summary = {
        "run_id": "r1",
        "status": "success",
        "mode": "single",
        "engine": "x",
        "receptor": "rec.pdb",
        "peptide": "pep.pdb",
        "runtime_sec": 1.2,
        "omega_full": 10,
        "omega_reduced": 3,
        "omega_ratio": 0.3,
        "n_pockets_total": 10,
        "n_pockets_selected": 3,
        "n_evals_cheap": 20,
        "n_evals_expensive": 2,
        "best_score_cheap": 1.0,
        "best_score_expensive": 1.1,
        "confidence_final": 0.8,
        "trigger_count_expensive": 2,
    }
    bundle = build_bundle(summary, [], None)
    assert bundle.summary.run_id == "r1"
