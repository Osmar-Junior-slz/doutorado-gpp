from dockingpp.reducao.modelos import (
    EntradaRankingBolsao,
    ResultadoAdmissibilidadeBolsao,
    ResultadoPreAfinidadeBolsao,
)
from dockingpp.reducao.ranking_bolsoes import ranquear_bolsoes_candidatos


def test_ranking_apenas_admissibilidade():
    resultados = [
        ResultadoAdmissibilidadeBolsao("b1", True, 0.8, ()),
        ResultadoAdmissibilidadeBolsao("b2", True, 0.4, ()),
    ]

    ranking = ranquear_bolsoes_candidatos(resultados)

    assert [item.id_bolsao for item in ranking] == ["b1", "b2"]
    assert ranking[0].score_pre_afinidade == 0.0


def test_ranking_com_admissibilidade_e_pre_afinidade():
    resultados_geo = [
        ResultadoAdmissibilidadeBolsao("b1", True, 0.6, ()),
        ResultadoAdmissibilidadeBolsao("b2", True, 0.6, ()),
    ]
    resultados_pre = [
        ResultadoPreAfinidadeBolsao("b1", 0.2, 0.0, 0.0, 0.0),
        ResultadoPreAfinidadeBolsao("b2", 0.9, 0.0, 0.0, 0.0),
    ]

    ranking = ranquear_bolsoes_candidatos(resultados_geo, resultados_pre)

    assert ranking[0].id_bolsao == "b2"
    assert ranking[1].id_bolsao == "b1"


def test_ranking_ordenacao_correta_maior_para_menor():
    resultados = [
        ResultadoAdmissibilidadeBolsao("b1", True, 0.2, ()),
        ResultadoAdmissibilidadeBolsao("b2", True, 0.9, ()),
        ResultadoAdmissibilidadeBolsao("b3", True, 0.6, ()),
    ]

    ranking = ranquear_bolsoes_candidatos(resultados)

    assert [item.id_bolsao for item in ranking] == ["b2", "b3", "b1"]


def test_nao_admissiveis_ficam_abaixo_dos_admissiveis():
    resultados = [
        ResultadoAdmissibilidadeBolsao("b1", False, 1.0, ("x",)),
        ResultadoAdmissibilidadeBolsao("b2", True, 0.2, ()),
        ResultadoAdmissibilidadeBolsao("b3", False, 0.9, ("x",)),
    ]

    ranking = ranquear_bolsoes_candidatos(resultados)

    assert ranking[0].admissivel is True
    assert ranking[1].admissivel is False
    assert ranking[2].admissivel is False


def test_retorno_tipado_entrada_ranking_bolsao():
    resultados = [ResultadoAdmissibilidadeBolsao("b1", True, 0.5, ())]

    ranking = ranquear_bolsoes_candidatos(resultados)

    assert isinstance(ranking, list)
    assert isinstance(ranking[0], EntradaRankingBolsao)
