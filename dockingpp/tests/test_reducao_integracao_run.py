import json

import numpy as np

from dockingpp.data.structs import Pocket, Pose, RunResult
from dockingpp.pipeline import run as run_module
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.reducao.modelos import (
    EntradaRankingBolsao,
    GeometriaBolsao,
    PerfilPeptideo,
    ResultadoAdmissibilidadeBolsao,
)


def _make_pockets(count: int) -> list[Pocket]:
    coords = np.zeros((1, 3), dtype=float)
    return [
        Pocket(id=f"pocket-{idx}", center=np.zeros(3, dtype=float), radius=5.0, coords=coords)
        for idx in range(count)
    ]


def _fake_rank_pockets(receptor, pockets, peptide=None, **_kwargs):
    return [(pocket, float(idx)) for idx, pocket in enumerate(pockets)]


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
    return RunResult(
        best_pose=Pose(
            coords=np.zeros((1, 3), dtype=float),
            score_cheap=0.0,
            score_expensive=0.0,
            meta={"generation": 0, "pocket_id": pockets[0].id if pockets else None},
        ),
        population=None,
    )


def test_flag_desligada_preserva_fluxo_reduced_antigo(monkeypatch, tmp_path):
    pockets = _make_pockets(5)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"coords": np.zeros((1, 3), dtype=float)}

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search)

    def _nao_deve_ser_chamado(*args, **kwargs):
        raise AssertionError("Subpipeline novo não deveria ser chamado com a flag desligada.")

    monkeypatch.setattr(run_module, "construir_perfil_peptideo", _nao_deve_ser_chamado)

    cfg = Config()
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.usar_reducao_condicionada_ao_peptideo = False

    out_dir = tmp_path / "reduced_sem_reducao_condicionada"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_pockets_used"] == 3


def test_flag_ligada_chama_cadeia_nova_de_reducao(monkeypatch, tmp_path):
    pockets = _make_pockets(3)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"coords": np.array([[0.0, 0.0, 0.0]], dtype=float)}
    chamadas = {
        "perfil": 0,
        "geometria": 0,
        "admissibilidade": 0,
        "ranking": 0,
        "seletor": 0,
        "pre_afinidade": 0,
    }

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search)

    def _perfil(_peptideo):
        chamadas["perfil"] += 1
        return PerfilPeptideo(1.0, 1.0, 1.0, 1.0, 1.0, 0.1)

    def _geometria(pocket):
        chamadas["geometria"] += 1
        return GeometriaBolsao(str(pocket.id), 1.0, 1.0, 1.0, 0.8, 0.2, 1.0)

    def _admissibilidade(_perfil_peptideo, geometria):
        chamadas["admissibilidade"] += 1
        return ResultadoAdmissibilidadeBolsao(geometria.id_bolsao, True, 0.9, ())


    def _pre_afinidade(_perfil_peptideo, geometria, score_encaixe_geometrico=None):
        chamadas["pre_afinidade"] += 1
        from dockingpp.reducao.modelos import ResultadoPreAfinidadeBolsao

        return ResultadoPreAfinidadeBolsao(geometria.id_bolsao, 0.5, 0.5, 0.1, 0.6)

    def _ranking(resultados_admissibilidade, resultados_pre_afinidade=None):
        chamadas["ranking"] += 1
        return [
            EntradaRankingBolsao(item.id_bolsao, item.admissivel, item.score_encaixe_geometrico, 0.0, item.score_encaixe_geometrico)
            for item in resultados_admissibilidade
        ]

    def _seletor(ranking_bolsoes, **kwargs):
        chamadas["seletor"] += 1
        return ranking_bolsoes[:1]

    monkeypatch.setattr(run_module, "construir_perfil_peptideo", _perfil)
    monkeypatch.setattr(run_module, "descrever_geometria_bolsao", _geometria)
    monkeypatch.setattr(run_module, "avaliar_admissibilidade_bolsao", _admissibilidade)
    monkeypatch.setattr(run_module, "estimar_pre_afinidade_bolsao", _pre_afinidade)
    monkeypatch.setattr(run_module, "ranquear_bolsoes_candidatos", _ranking)
    monkeypatch.setattr(run_module, "selecionar_bolsoes_para_busca", _seletor)

    cfg = Config()
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.usar_reducao_condicionada_ao_peptideo = True

    out_dir = tmp_path / "reduced_com_reducao_condicionada"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert chamadas["perfil"] == 1
    assert chamadas["geometria"] == 3
    assert chamadas["admissibilidade"] == 3
    assert chamadas["ranking"] == 1
    assert chamadas["seletor"] == 1
    assert chamadas["pre_afinidade"] == 3
    assert summary["n_pockets_used"] == 1


def test_fallback_antigo_continua_funcionando_com_flag_ligada(monkeypatch, tmp_path):
    pockets = _make_pockets(2)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"coords": np.array([[0.0, 0.0, 0.0]], dtype=float)}

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search)
    monkeypatch.setattr(
        run_module,
        "selecionar_bolsoes_para_busca",
        lambda ranking_bolsoes, **kwargs: [],
    )

    cfg = Config()
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 2
    cfg.usar_reducao_condicionada_ao_peptideo = True
    cfg.reducao = {"permitir_fallback": False}

    out_dir = tmp_path / "reduced_com_fallback"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["fallback_to_full"] is True
    assert (out_dir / "fallback_full" / "result.json").exists()


def test_fluxo_full_nao_chama_reducao_condicionada(monkeypatch, tmp_path):
    pockets = _make_pockets(3)
    receptor = {"coords": np.zeros((1, 3), dtype=float)}
    peptide = {"coords": np.array([[0.0, 0.0, 0.0]], dtype=float)}

    monkeypatch.setattr(run_module, "_dummy_inputs", lambda: (receptor, peptide, pockets))
    monkeypatch.setattr(run_module, "rank_pockets", _fake_rank_pockets)
    monkeypatch.setattr(run_module.ABCGAVGOSSearch, "search", _fake_search)

    def _nao_deve_ser_chamado(*args, **kwargs):
        raise AssertionError("Fluxo full não deve chamar redução condicionada.")

    monkeypatch.setattr(run_module, "construir_perfil_peptideo", _nao_deve_ser_chamado)

    cfg = Config()
    cfg.search_space_mode = "full"
    cfg.full_search = True
    cfg.usar_reducao_condicionada_ao_peptideo = True

    out_dir = tmp_path / "full_sem_reducao_condicionada"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["search_space_mode"] == "full"
