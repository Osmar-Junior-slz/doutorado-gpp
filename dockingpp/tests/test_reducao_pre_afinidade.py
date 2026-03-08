import math

from dockingpp.reducao.modelos import GeometriaBolsao, PerfilPeptideo, ResultadoAdmissibilidadeBolsao, ResultadoPreAfinidadeBolsao
from dockingpp.reducao.pre_afinidade import estimar_pre_afinidade_bolsao
from dockingpp.reducao.ranking_bolsoes import ranquear_bolsoes_candidatos


def _perfil_base() -> PerfilPeptideo:
    return PerfilPeptideo(
        comprimento_efetivo=10.0,
        largura_efetiva=4.0,
        espessura_efetiva=2.0,
        extensao_maxima=11.0,
        raio_giro=3.2,
        indice_flexibilidade=0.25,
    )


def test_estimar_pre_afinidade_cria_resultado_tipado():
    resultado = estimar_pre_afinidade_bolsao(
        _perfil_base(),
        GeometriaBolsao("b1", 9.0, 3.5, 2.0, 0.7, 0.4, 60.0),
    )

    assert isinstance(resultado, ResultadoPreAfinidadeBolsao)
    assert resultado.id_bolsao == "b1"


def test_estimar_pre_afinidade_score_coerente_caso_simples():
    perfil = _perfil_base()
    bom = estimar_pre_afinidade_bolsao(perfil, GeometriaBolsao("bom", 10.0, 4.0, 2.0, 0.9, 0.3, 80.0), 0.9)
    ruim = estimar_pre_afinidade_bolsao(perfil, GeometriaBolsao("ruim", 4.0, 1.0, 0.5, 0.2, 1.0, 5.0), 0.2)

    assert bom.score_pre_afinidade > ruim.score_pre_afinidade
    assert 0.0 <= bom.score_pre_afinidade <= 1.0


def test_estimar_pre_afinidade_estavel_com_dados_pobres():
    resultado = estimar_pre_afinidade_bolsao(
        _perfil_base(),
        GeometriaBolsao("vazio", 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )

    assert math.isfinite(resultado.score_pre_afinidade)
    assert 0.0 <= resultado.score_pre_afinidade <= 1.0
    assert 0.0 <= resultado.score_contatos <= 1.0
    assert 0.0 <= resultado.penalidade_clash <= 1.0
    assert 0.0 <= resultado.score_ancoragem <= 1.0


def test_integracao_ranking_considera_pre_afinidade_quando_fornecida():
    admissibilidade = [
        ResultadoAdmissibilidadeBolsao("b1", True, 0.6, ()),
        ResultadoAdmissibilidadeBolsao("b2", True, 0.6, ()),
    ]
    pre_afinidade = [
        ResultadoPreAfinidadeBolsao("b1", 0.1, 0.0, 0.0, 0.0),
        ResultadoPreAfinidadeBolsao("b2", 0.9, 0.0, 0.0, 0.0),
    ]

    ranking = ranquear_bolsoes_candidatos(admissibilidade, pre_afinidade)

    assert [item.id_bolsao for item in ranking] == ["b2", "b1"]
