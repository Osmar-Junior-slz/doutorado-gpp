from __future__ import annotations

import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.pipeline.execucao.selecao_bolsoes import SelecionadorBolsoesPipeline
from dockingpp.pipeline.run import Config, _cfg_value, _extract_coords, _normalize_search_space_mode


class _Tracer:
    def __init__(self) -> None:
        self.search_space_mode = None
        self.events: list[dict[str, object]] = []

    def event(self, **kwargs):
        self.events.append(kwargs)


class _DebugLogger:
    def __init__(self) -> None:
        self.logs: list[dict[str, object]] = []

    def log(self, payload):
        self.logs.append(payload)


def _make_pockets(count: int) -> list[Pocket]:
    coords = np.zeros((1, 3), dtype=float)
    return [
        Pocket(id=f"pocket-{idx}", center=np.zeros(3, dtype=float), radius=5.0, coords=coords)
        for idx in range(count)
    ]


def _make_selector() -> SelecionadorBolsoesPipeline:
    return SelecionadorBolsoesPipeline(
        normalizar_modo_busca=_normalize_search_space_mode,
        obter_valor_cfg=_cfg_value,
        extrair_coords=_extract_coords,
        aplicar_reducao_condicionada=lambda peptide, pockets, cfg, tracer, debug_logger: pockets,
    )


def test_selector_exposes_candidate_selected_and_accepted(monkeypatch) -> None:
    pockets = _make_pockets(3)
    tracer = _Tracer()
    debug_logger = _DebugLogger()

    monkeypatch.setattr("dockingpp.pipeline.execucao.selecao_bolsoes.rank_pockets", lambda receptor, pockets, **_: [(p, float(idx)) for idx, p in enumerate(pockets)])

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

    contexto = _make_selector().selecionar(
        cfg=cfg,
        receptor={"coords": np.zeros((1, 3), dtype=float)},
        peptide={"coords": np.zeros((1, 3), dtype=float)},
        dummy_pockets=pockets,
        tracer=tracer,
        debug_logger=debug_logger,
    )

    assert len(contexto.candidate_pockets) == 3
    assert [str(p.id) for p in contexto.selected_pockets] == ["pocket-0", "pocket-1", "pocket-2"]
    assert [str(p.id) for _, p in contexto.accepted_pockets] == ["pocket-0", "pocket-2"]
    assert contexto.feasible_pockets == contexto.accepted_pockets
    assert contexto.pockets == contexto.selected_pockets
    assert any(item["reason"] == "feasible_fraction<=0.0" for item in contexto.rejected)
    assert any(evt.get("event_type") == "pocket_acceptance_summary" for evt in tracer.events)


def test_selector_full_mode_keeps_equivalent_behavior() -> None:
    tracer = _Tracer()
    debug_logger = _DebugLogger()

    cfg = Config()
    cfg.search_space_mode = "full"
    cfg.full_search = True
    cfg.scan = {"enabled": True}

    contexto = _make_selector().selecionar(
        cfg=cfg,
        receptor={"coords": np.zeros((1, 3), dtype=float)},
        peptide={"coords": np.zeros((1, 3), dtype=float)},
        dummy_pockets=_make_pockets(4),
        tracer=tracer,
        debug_logger=debug_logger,
    )

    assert contexto.search_space_mode == "full"
    assert contexto.total_pockets == 1
    assert len(contexto.candidate_pockets) == 1
    assert len(contexto.selected_pockets) == 1
    assert len(contexto.accepted_pockets) == 1
    assert contexto.scan_params["enabled"] is False


def test_selector_legacy_mode_remains_compatible(monkeypatch) -> None:
    pockets = _make_pockets(4)
    tracer = _Tracer()
    debug_logger = _DebugLogger()

    monkeypatch.setattr("dockingpp.pipeline.execucao.selecao_bolsoes.rank_pockets", lambda receptor, pockets, **_: [(p, float(idx)) for idx, p in enumerate(pockets)])

    cfg = Config()
    cfg.search_space_mode = "pockets"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.scan = {"enabled": False}

    contexto = _make_selector().selecionar(
        cfg=cfg,
        receptor={"coords": np.zeros((1, 3), dtype=float)},
        peptide={"coords": np.zeros((1, 3), dtype=float)},
        dummy_pockets=pockets,
        tracer=tracer,
        debug_logger=debug_logger,
    )

    assert contexto.modo_legado_pockets is True
    assert contexto.search_space_mode == "reduced"
    assert len(contexto.candidate_pockets) == 2
    assert len(contexto.selected_pockets) == 2
    assert len(contexto.accepted_pockets) == 2
    assert not contexto.rejected
