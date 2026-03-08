from dockingpp.reducao.admissibilidade import avaliar_admissibilidade_bolsao
from dockingpp.reducao.modelos import GeometriaBolsao, PerfilPeptideo


def _perfil_base() -> PerfilPeptideo:
    return PerfilPeptideo(
        comprimento_efetivo=10.0,
        largura_efetiva=4.0,
        espessura_efetiva=2.0,
        extensao_maxima=10.5,
        raio_giro=3.0,
        indice_flexibilidade=0.2,
    )


def test_admissibilidade_reprova_bolsao_pequeno_demais():
    resultado = avaliar_admissibilidade_bolsao(
        _perfil_base(),
        GeometriaBolsao(
            id_bolsao="b1",
            comprimento_util=6.0,
            largura_util=4.0,
            profundidade_util=3.0,
            continuidade_superficial=0.6,
            exposicao_superficial=0.4,
            volume_estimado=40.0,
        ),
    )

    assert resultado.admissivel is False
    assert "comprimento_util_insuficiente" in resultado.motivos_reprovacao


def test_admissibilidade_reprova_largura_insuficiente():
    resultado = avaliar_admissibilidade_bolsao(
        _perfil_base(),
        GeometriaBolsao(
            id_bolsao="b2",
            comprimento_util=10.0,
            largura_util=2.0,
            profundidade_util=3.0,
            continuidade_superficial=0.7,
            exposicao_superficial=0.3,
            volume_estimado=60.0,
        ),
    )

    assert resultado.admissivel is False
    assert "largura_util_insuficiente" in resultado.motivos_reprovacao


def test_admissibilidade_aprova_bolsao_razoavel():
    resultado = avaliar_admissibilidade_bolsao(
        _perfil_base(),
        GeometriaBolsao(
            id_bolsao="b3",
            comprimento_util=9.0,
            largura_util=3.2,
            profundidade_util=3.0,
            continuidade_superficial=0.65,
            exposicao_superficial=0.55,
            volume_estimado=75.0,
        ),
    )

    assert resultado.admissivel is True
    assert resultado.motivos_reprovacao == ()


def test_admissibilidade_score_no_intervalo_esperado():
    resultado = avaliar_admissibilidade_bolsao(
        _perfil_base(),
        GeometriaBolsao(
            id_bolsao="b4",
            comprimento_util=20.0,
            largura_util=10.0,
            profundidade_util=5.0,
            continuidade_superficial=1.0,
            exposicao_superficial=1.0,
            volume_estimado=100.0,
        ),
    )

    assert 0.0 <= resultado.score_encaixe_geometrico <= 1.0


def test_admissibilidade_preenche_multiplos_motivos_reprovacao():
    resultado = avaliar_admissibilidade_bolsao(
        _perfil_base(),
        GeometriaBolsao(
            id_bolsao="b5",
            comprimento_util=5.0,
            largura_util=1.0,
            profundidade_util=2.0,
            continuidade_superficial=0.1,
            exposicao_superficial=0.99,
            volume_estimado=10.0,
        ),
    )

    assert resultado.admissivel is False
    assert "comprimento_util_insuficiente" in resultado.motivos_reprovacao
    assert "largura_util_insuficiente" in resultado.motivos_reprovacao
    assert "continuidade_superficial_baixa" in resultado.motivos_reprovacao
    assert "exposicao_superficial_excessiva" in resultado.motivos_reprovacao
