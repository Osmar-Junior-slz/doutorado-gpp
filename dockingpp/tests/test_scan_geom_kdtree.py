import json

import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.pipeline.scan import (
    build_receptor_kdtree,
    scan_pocket_feasibility_geom_kdtree,
    select_pockets_from_scan,
)


def _geom_cfg(**overrides):
    cfg = {
        "selector_mode": "geom_kdtree",
        "samples_per_pocket": 24,
        "translation_sigma": 1.5,
        "rotation_max_deg": 25.0,
        "severe_clash_threshold": 0.6,
        "moderate_clash_threshold": 1.1,
        "contact_min": 1.0,
        "contact_max": 2.8,
        "exposure_margin": 1.0,
        "reject_if_feasible_fraction_leq": 0.0,
        "reject_if_severe_clash_fraction_geq": 0.95,
        "score_alpha": 1.0,
        "score_beta": 0.5,
        "score_gamma": 2.0,
        "score_delta": 0.25,
    }
    cfg.update(overrides)
    return cfg


def test_geom_scan_generates_metrics_per_pocket():
    receptor_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    peptide_coords = np.array([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.1, 0.0]], dtype=float)
    pocket = Pocket(id="p0", center=np.array([0.5, 0.5, 0.0]), radius=2.0, coords=receptor_coords)

    rng = np.random.default_rng(10)
    metrics = scan_pocket_feasibility_geom_kdtree(
        build_receptor_kdtree(receptor_coords),
        receptor_coords,
        peptide_coords,
        pocket,
        _geom_cfg(),
        rng,
    )

    required = {
        "feasible_fraction",
        "severe_clash_fraction",
        "moderate_clash_mean",
        "mean_contacts",
        "best_contact_count",
        "best_geom_energy",
        "mean_exposure_penalty",
        "pocket_scan_score",
        "n_samples",
        "best_sample_meta",
    }
    assert required.issubset(metrics.keys())


def test_geom_scan_ranks_pockets_by_score():
    receptor_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.2, 0.0], [1.2, 0.0, 0.0]], dtype=float)
    peptide_coords = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]], dtype=float)
    p_good = Pocket(id="good", center=np.array([0.6, 0.6, 0.0]), radius=2.0, coords=receptor_coords)
    p_bad = Pocket(id="bad", center=np.array([8.0, 8.0, 8.0]), radius=2.0, coords=receptor_coords)

    rng = np.random.default_rng(22)
    tree = build_receptor_kdtree(receptor_coords)
    table = {
        "good": scan_pocket_feasibility_geom_kdtree(tree, receptor_coords, peptide_coords, p_good, _geom_cfg(), rng),
        "bad": scan_pocket_feasibility_geom_kdtree(tree, receptor_coords, peptide_coords, p_bad, _geom_cfg(), rng),
    }

    selected = select_pockets_from_scan([p_good, p_bad], table, top_k=1, selector_mode="geom_kdtree")
    assert selected[0].id == "good"


def test_geom_scan_rejects_only_clearly_inviable_pockets(tmp_path):
    cfg = Config(seed=1, generations=1, pop_size=2, topk=1)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.scan = _geom_cfg(enabled=True, select_top_k=3, samples_per_pocket=16, severe_clash_threshold=0.1)

    out = tmp_path / "reduced_geom"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["mode"] == "reduced_aggregate"
    assert summary["n_pockets_used"] >= 1


def test_reduced_uses_geom_scan_when_configured(tmp_path):
    cfg = Config(seed=2, generations=1, pop_size=2, topk=1)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.scan = _geom_cfg(enabled=True, select_top_k=2)

    out = tmp_path / "reduced_geom_mode"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))

    selected = json.loads((out / "summary.json").read_text(encoding="utf-8"))["selected_pockets"]
    first = json.loads((out / selected[0] / "summary.json").read_text(encoding="utf-8"))

    assert first["scan"]["selector_mode"] == "geom_kdtree"


def test_reduced_fallback_still_works_when_no_pocket_passes(tmp_path):
    cfg = Config(seed=3, generations=1, pop_size=2, topk=1)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.scan = _geom_cfg(
        enabled=True,
        select_top_k=3,
        reject_if_feasible_fraction_leq=1.0,
        reject_if_severe_clash_fraction_geq=0.0,
    )

    out = tmp_path / "reduced_geom_fallback"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["fallback_to_full"] is True


def test_full_mode_is_unchanged(tmp_path):
    cfg = Config(seed=4, generations=1, pop_size=2, topk=1)
    cfg.search_space_mode = "full"
    cfg.full_search = True
    cfg.scan = _geom_cfg(enabled=True)

    out = tmp_path / "full_unchanged"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["search_space_mode"] == "full"
    assert summary["scan"]["enabled"] is False


def test_scan_by_pocket_contains_new_metrics(tmp_path):
    cfg = Config(seed=5, generations=1, pop_size=2, topk=1)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.scan = _geom_cfg(enabled=True, select_top_k=2)

    out = tmp_path / "reduced_geom_metrics"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))

    selected = json.loads((out / "summary.json").read_text(encoding="utf-8"))["selected_pockets"]
    first = json.loads((out / selected[0] / "summary.json").read_text(encoding="utf-8"))
    metrics = next(iter(first["scan_by_pocket"].values()))

    assert "pocket_scan_score" in metrics
    assert "best_geom_energy" in metrics


def test_legacy_mode_still_works(tmp_path):
    cfg = Config(seed=6, generations=1, pop_size=2, topk=1)
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.scan = {"enabled": True, "selector_mode": "legacy", "select_top_k": 2, "max_clash_ratio": 1.0, "samples_per_pocket": 8}

    out = tmp_path / "reduced_legacy"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["mode"] == "reduced_aggregate"
    assert summary["n_pockets_used"] >= 1


def test_pocket_scan_score_uses_energy_to_score_conversion_consistently():
    receptor_coords = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0]], dtype=float)
    peptide_coords = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]], dtype=float)
    pocket = Pocket(id="p1", center=np.array([0.6, 0.6, 0.0]), radius=2.0, coords=receptor_coords)

    cfg = _geom_cfg(score_alpha=0.0, score_beta=0.0, score_gamma=0.0, score_delta=0.0)
    metrics = scan_pocket_feasibility_geom_kdtree(
        build_receptor_kdtree(receptor_coords), receptor_coords, peptide_coords, pocket, cfg, np.random.default_rng(7)
    )

    assert np.isclose(metrics["pocket_scan_score"], -metrics["best_geom_energy"])


def test_debug_trace_records_geom_scan_selection(tmp_path):
    cfg = Config(seed=8, generations=1, pop_size=2, topk=1)
    cfg.debug_enabled = True
    cfg.debug_level = "AUDIT"
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.scan = _geom_cfg(enabled=True, select_top_k=2)

    out = tmp_path / "reduced_geom_trace"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out))

    lines = (out / "debug" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    matches = [json.loads(line) for line in lines if json.loads(line).get("event_type") == "scan_selection_ranked"]
    assert matches
    assert matches[-1]["payload"]["selector_mode"] == "geom_kdtree"
