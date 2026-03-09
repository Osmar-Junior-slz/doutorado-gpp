import json

import numpy as np

from dockingpp.data.structs import Pocket, Pose, RunResult
from dockingpp.pipeline import run as run_module
from dockingpp.pipeline.run import Config, run_pipeline


def _make_pockets(count):
    coords = np.zeros((1, 3), dtype=float)
    return [
        Pocket(id=f"pocket-{idx}", center=np.zeros(3, dtype=float), radius=5.0, coords=coords)
        for idx in range(count)
    ]


def _fake_rank_pockets(receptor, pockets, peptide=None, **_kwargs):
    return [(pocket, float(idx)) for idx, pocket in enumerate(pockets)]


def _fake_search_factory(captured):
    def _fake_search(
        self,
        receptor,
        peptide,
        pockets,
        cfg,
        score_cheap_fn,
        score_expensive_fn,
        prior_pocket,
        prior_pose,
        logger,
    ):
        captured["pockets"] = pockets
        return RunResult(
            best_pose=Pose(
                coords=np.zeros((1, 3), dtype=float),
                score_cheap=0.0,
                score_expensive=0.0,
                meta={"generation": 0, "pocket_id": pockets[0].id if pockets else None},
            ),
            population=None,
        )

    return _fake_search


def _fake_search_factory_collect_calls(captured):
    def _fake_search(
        self,
        receptor,
        peptide,
        pockets,
        cfg,
        score_cheap_fn,
        score_expensive_fn,
        prior_pocket,
        prior_pose,
        logger,
    ):
        captured.setdefault("calls", []).append([str(p.id) for p in pockets])
        return RunResult(
            best_pose=Pose(
                coords=np.zeros((1, 3), dtype=float),
                score_cheap=0.0,
                score_expensive=0.0,
                meta={"generation": 0, "pocket_id": pockets[0].id if pockets else None},
            ),
            population=None,
        )

    return _fake_search


def test_full_search_applies_max_pockets_used(monkeypatch, tmp_path):
    pockets = _make_pockets(20)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"dummy": True}
    captured = {}

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search_factory(captured))

    cfg = Config()
    cfg.full_search = True
    cfg.max_pockets_used = 5
    cfg.search_space_mode = "pockets"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(tmp_path))

    assert len(captured["pockets"]) == 5


def test_reduced_search_still_uses_top_pockets(monkeypatch, tmp_path):
    pockets = _make_pockets(20)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"dummy": True}
    captured = {}

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search_factory(captured))

    cfg = Config()
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.max_pockets_used = 8
    cfg.search_space_mode = "pockets"

    run_pipeline(cfg, "__dummy__", "__dummy__", str(tmp_path))

    assert len(captured["pockets"]) == 3


def test_reduced_aggregate_executes_only_accepted_pockets(monkeypatch, tmp_path):
    pockets = _make_pockets(3)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"coords": np.zeros((1, 3), dtype=float)}
    captured = {}

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search_factory_collect_calls(captured))

    def _scan_with_one_rejection(_tree, _peptide_coords, pocket, _scan_cfg, _rng):
        if str(pocket.id) == "pocket-1":
            return {"feasible_fraction": 0.0, "clash_ratio_best": 0.0, "scan_score": 0.1}
        return {"feasible_fraction": 1.0, "clash_ratio_best": 0.0, "scan_score": 1.0}

    monkeypatch.setattr("dockingpp.pipeline.execucao.selecao_bolsoes.scan_pocket_feasibility", _scan_with_one_rejection)

    cfg = Config()
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.scan = {
        "enabled": True,
        "selector_mode": "legacy",
        "select_top_k": 3,
        "max_clash_ratio": 1.0,
        "seed_offset": 0,
    }

    out_dir = tmp_path / "reduced_acceptance"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["mode"] == "reduced_aggregate"
    assert summary["n_pockets_used"] == 2
    assert len(summary["selected_pockets"]) == 2
    assert summary["selected_pre_filter_pockets"] == ["pocket-0", "pocket-1", "pocket-2"]
    assert any(item["reason"] == "feasible_fraction<=0.0" for item in summary["rejected_pockets"])

    calls = captured.get("calls", [])
    assert len(calls) == 2
    executed = sorted(call[0] for call in calls if call)
    assert executed == ["pocket-0", "pocket-2"]
