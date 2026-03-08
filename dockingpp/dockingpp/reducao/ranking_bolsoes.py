"""Ranking reutilizável de bolsões com sinais geométricos e de pré-afinidade."""

from __future__ import annotations

from typing import Iterable

from dockingpp.reducao.modelos import (
    EntradaRankingBolsao,
    ResultadoAdmissibilidadeBolsao,
    ResultadoPreAfinidadeBolsao,
)


def _normalizar_score(score: float) -> float:
    """Normaliza score para intervalo [0, 1] por truncamento simples."""

    return max(0.0, min(1.0, float(score)))


def _combinar_scores(
    score_encaixe_geometrico: float,
    score_pre_afinidade: float,
    admissivel: bool,
) -> float:
    """Combina sinais de geometria e pré-afinidade em score final.

    Nesta versão inicial, a geometria recebe peso maior para priorizar bolsões
    plausíveis no filtro de redução.
    """

    score_geo = _normalizar_score(score_encaixe_geometrico)
    score_pre = _normalizar_score(score_pre_afinidade)

    score_base = (0.75 * score_geo) + (0.25 * score_pre)

    # Regra de ordenação: não admissíveis devem ficar naturalmente abaixo.
    if not admissivel:
        score_base *= 0.5

    return _normalizar_score(score_base)


def _mapa_pre_afinidade(
    resultados_pre_afinidade: Iterable[ResultadoPreAfinidadeBolsao] | None,
) -> dict[str, ResultadoPreAfinidadeBolsao]:
    """Indexa resultados de pré-afinidade por `id_bolsao`."""

    if resultados_pre_afinidade is None:
        return {}
    return {resultado.id_bolsao: resultado for resultado in resultados_pre_afinidade}


def ranquear_bolsoes_candidatos(
    resultados_admissibilidade: list[ResultadoAdmissibilidadeBolsao],
    resultados_pre_afinidade: list[ResultadoPreAfinidadeBolsao] | None = None,
) -> list[EntradaRankingBolsao]:
    """Ranqueia bolsões candidatos do melhor para o pior.

    O ranking funciona mesmo sem pré-afinidade, usando apenas o score geométrico.
    """

    mapa_pre = _mapa_pre_afinidade(resultados_pre_afinidade)
    entradas: list[EntradaRankingBolsao] = []

    for resultado_geo in resultados_admissibilidade:
        pre = mapa_pre.get(resultado_geo.id_bolsao)
        score_pre = float(pre.score_pre_afinidade) if pre is not None else 0.0
        score_final = _combinar_scores(
            score_encaixe_geometrico=resultado_geo.score_encaixe_geometrico,
            score_pre_afinidade=score_pre,
            admissivel=resultado_geo.admissivel,
        )

        entradas.append(
            EntradaRankingBolsao(
                id_bolsao=resultado_geo.id_bolsao,
                admissivel=resultado_geo.admissivel,
                score_encaixe_geometrico=float(resultado_geo.score_encaixe_geometrico),
                score_pre_afinidade=score_pre,
                score_final=score_final,
            )
        )

    entradas.sort(key=lambda item: (item.admissivel, item.score_final), reverse=True)
    return entradas
