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
                meta={"generation": 0},
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
