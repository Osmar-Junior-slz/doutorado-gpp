"""Seleção de bolsões candidatos para a etapa de busca detalhada."""

from __future__ import annotations

from dockingpp.reducao.modelos import EntradaRankingBolsao


def _aplicar_filtros(
    ranking: list[EntradaRankingBolsao],
    *,
    apenas_admissiveis: bool,
    score_minimo: float | None,
) -> list[EntradaRankingBolsao]:
    """Aplica filtros de elegibilidade preservando a ordem recebida."""

    selecionados = ranking
    if apenas_admissiveis:
        selecionados = [item for item in selecionados if item.admissivel]
    if score_minimo is not None:
        limiar = float(score_minimo)
        selecionados = [item for item in selecionados if item.score_final >= limiar]
    return selecionados


def _aplicar_top_k(
    ranking: list[EntradaRankingBolsao],
    top_k: int | None,
) -> list[EntradaRankingBolsao]:
    """Limita quantidade de itens sem alterar a ordenação."""

    if top_k is None:
        return ranking
    limite = max(0, int(top_k))
    return ranking[:limite]


def selecionar_bolsoes_para_busca(
    ranking_bolsoes: list[EntradaRankingBolsao],
    *,
    top_k: int | None = None,
    score_minimo: float | None = None,
    apenas_admissiveis: bool = False,
    permitir_fallback: bool = False,
    quantidade_fallback: int = 1,
) -> list[EntradaRankingBolsao]:
    """Seleciona bolsões para busca detalhada com política de fallback.

    Política de fallback: se, após os filtros, nada for selecionado e
    `permitir_fallback=True`, retorna os melhores itens do ranking original
    (sem filtros) limitados por `quantidade_fallback`.
    """

    selecionados = _aplicar_filtros(
        ranking_bolsoes,
        apenas_admissiveis=apenas_admissiveis,
        score_minimo=score_minimo,
    )
    selecionados = _aplicar_top_k(selecionados, top_k)

    if selecionados:
        return selecionados
    if not permitir_fallback:
        return []

    # Fallback controlado: usa o topo do ranking original sem refiltrar.
    limite_fallback = max(0, int(quantidade_fallback))
    return ranking_bolsoes[:limite_fallback]
