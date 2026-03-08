from dockingpp.reducao.modelos import EntradaRankingBolsao
from dockingpp.reducao.seletor_bolsoes import selecionar_bolsoes_para_busca


def _ranking_base() -> list[EntradaRankingBolsao]:
    return [
        EntradaRankingBolsao("b1", True, 0.9, 0.2, 0.8),
        EntradaRankingBolsao("b2", False, 0.95, 0.9, 0.7),
        EntradaRankingBolsao("b3", True, 0.6, 0.1, 0.5),
        EntradaRankingBolsao("b4", True, 0.4, 0.0, 0.3),
    ]


def test_seletor_por_top_k():
    selecionados = selecionar_bolsoes_para_busca(_ranking_base(), top_k=2)

    assert [item.id_bolsao for item in selecionados] == ["b1", "b2"]


def test_seletor_por_score_minimo():
    selecionados = selecionar_bolsoes_para_busca(_ranking_base(), score_minimo=0.6)

    assert [item.id_bolsao for item in selecionados] == ["b1", "b2"]


def test_seletor_exclui_nao_admissiveis():
    selecionados = selecionar_bolsoes_para_busca(_ranking_base(), apenas_admissiveis=True)

    assert [item.id_bolsao for item in selecionados] == ["b1", "b3", "b4"]
    assert all(item.admissivel for item in selecionados)


def test_seletor_fallback_quando_lista_vazia():
    selecionados = selecionar_bolsoes_para_busca(
        _ranking_base(),
        score_minimo=0.99,
        permitir_fallback=True,
        quantidade_fallback=1,
    )

    assert [item.id_bolsao for item in selecionados] == ["b1"]


def test_seletor_preserva_ordenacao_do_ranking_filtrado():
    selecionados = selecionar_bolsoes_para_busca(
        _ranking_base(),
        apenas_admissiveis=True,
        score_minimo=0.3,
    )

    assert [item.id_bolsao for item in selecionados] == ["b1", "b3", "b4"]
