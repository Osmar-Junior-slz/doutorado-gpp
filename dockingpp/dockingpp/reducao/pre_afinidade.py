"""Estimativa barata e preliminar de pré-afinidade para bolsões.

Este módulo NÃO substitui o score final de docking. A intenção é apenas gerar
um sinal barato para ordenar bolsões já plausíveis geometricamente.
"""

from __future__ import annotations

import numpy as np

from dockingpp.reducao.modelos import GeometriaBolsao, PerfilPeptideo, ResultadoPreAfinidadeBolsao


def _normalizar_score(valor: float) -> float:
    """Normaliza valores no intervalo [0, 1]."""

    return float(np.clip(float(valor), 0.0, 1.0))


def _score_contatos(perfil_peptideo: PerfilPeptideo, geometria_bolsao: GeometriaBolsao) -> float:
    """Heurística de contatos baseada em compatibilidade de dimensões e volume."""

    alvo_comp = max(1e-6, float(perfil_peptideo.comprimento_efetivo))
    alvo_larg = max(1e-6, float(perfil_peptideo.largura_efetiva))
    alvo_prof = max(1e-6, float(perfil_peptideo.espessura_efetiva))
    volume_alvo = max(1e-6, alvo_comp * alvo_larg * alvo_prof)

    razao_comp = float(geometria_bolsao.comprimento_util) / alvo_comp
    razao_larg = float(geometria_bolsao.largura_util) / alvo_larg
    razao_prof = float(geometria_bolsao.profundidade_util) / alvo_prof
    razao_volume = float(geometria_bolsao.volume_estimado) / volume_alvo

    score_dimensional = np.mean([np.clip(razao_comp, 0.0, 1.0), np.clip(razao_larg, 0.0, 1.0), np.clip(razao_prof, 0.0, 1.0)])
    score_volume = float(np.clip(razao_volume, 0.0, 1.0))
    return _normalizar_score((0.7 * float(score_dimensional)) + (0.3 * score_volume))


def _penalidade_clash(perfil_peptideo: PerfilPeptideo, geometria_bolsao: GeometriaBolsao) -> float:
    """Penalidade barata de colisão por excesso de exposição e pouca profundidade."""

    exposicao = _normalizar_score(geometria_bolsao.exposicao_superficial)
    profundidade_alvo = max(1e-6, float(perfil_peptideo.espessura_efetiva))
    razao_profundidade = float(geometria_bolsao.profundidade_util) / profundidade_alvo

    penalidade_exposicao = max(0.0, exposicao - 0.6)
    penalidade_profundidade = max(0.0, 1.0 - float(np.clip(razao_profundidade, 0.0, 1.0)))
    return _normalizar_score((0.6 * penalidade_exposicao) + (0.4 * penalidade_profundidade))


def _score_ancoragem(geometria_bolsao: GeometriaBolsao) -> float:
    """Sinal de ancoragem com base em continuidade e exposição moderada."""

    continuidade = _normalizar_score(geometria_bolsao.continuidade_superficial)
    exposicao = _normalizar_score(geometria_bolsao.exposicao_superficial)
    # Preferimos continuidade alta com exposição não extrema.
    return _normalizar_score((0.7 * continuidade) + (0.3 * (1.0 - exposicao)))


def estimar_pre_afinidade_bolsao(
    perfil_peptideo: PerfilPeptideo,
    geometria_bolsao: GeometriaBolsao,
    score_encaixe_geometrico: float | None = None,
) -> ResultadoPreAfinidadeBolsao:
    """Estima pré-afinidade barata para priorização inicial de bolsões.

    O cálculo usa apenas heurísticas geométricas simples e opcionais de
    encaixe prévio, para manter custo computacional baixo.
    """

    score_contatos = _score_contatos(perfil_peptideo, geometria_bolsao)
    penalidade_clash = _penalidade_clash(perfil_peptideo, geometria_bolsao)
    score_ancoragem = _score_ancoragem(geometria_bolsao)

    score_base = (0.5 * score_contatos) + (0.35 * score_ancoragem) - (0.25 * penalidade_clash)
    if score_encaixe_geometrico is not None:
        # Usa encaixe geométrico como ajuste leve para reaproveitar sinal já existente.
        score_base = (0.85 * score_base) + (0.15 * _normalizar_score(score_encaixe_geometrico))

    return ResultadoPreAfinidadeBolsao(
        id_bolsao=geometria_bolsao.id_bolsao,
        score_pre_afinidade=_normalizar_score(score_base),
        score_contatos=score_contatos,
        penalidade_clash=penalidade_clash,
        score_ancoragem=score_ancoragem,
    )
