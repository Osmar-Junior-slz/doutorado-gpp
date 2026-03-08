"""Avaliação heurística de admissibilidade geométrica de bolsões."""

from __future__ import annotations

from dockingpp.reducao.modelos import (
    GeometriaBolsao,
    PerfilPeptideo,
    ResultadoAdmissibilidadeBolsao,
)


def _score_dimensao(valor_bolsao: float, valor_peptideo: float, fator_minimo: float) -> float:
    """Converte razão de dimensão em score no intervalo [0, 1].

    A regra privilegia bolsões que atendem ao tamanho mínimo relativo ao
    peptídeo, mas mantém gradação para apoiar ordenação posterior.
    """

    alvo = max(1e-6, float(valor_peptideo) * float(fator_minimo))
    razao = float(valor_bolsao) / alvo
    return max(0.0, min(1.0, razao))


def _penalidade_exposicao(exposicao_superficial: float) -> float:
    """Retorna penalidade por exposição excessiva.

    Exposição moderada é tolerada, enquanto valores muito altos reduzem o score
    por sugerirem uma região menos confinada.
    """

    exposicao = max(0.0, min(1.0, float(exposicao_superficial)))
    limite_tolerado = 0.75
    if exposicao <= limite_tolerado:
        return 0.0

    excesso = (exposicao - limite_tolerado) / (1.0 - limite_tolerado)
    return 0.2 * max(0.0, min(1.0, excesso))


def _calcular_score_encaixe(perfil_peptideo: PerfilPeptideo, geometria_bolsao: GeometriaBolsao) -> float:
    """Combina componentes geométricos em score de encaixe entre 0 e 1."""

    score_comprimento = _score_dimensao(
        geometria_bolsao.comprimento_util,
        perfil_peptideo.comprimento_efetivo,
        fator_minimo=0.80,
    )
    score_largura = _score_dimensao(
        geometria_bolsao.largura_util,
        perfil_peptideo.largura_efetiva,
        fator_minimo=0.70,
    )
    score_continuidade = max(0.0, min(1.0, float(geometria_bolsao.continuidade_superficial)))

    score_base = (0.45 * score_comprimento) + (0.35 * score_largura) + (0.20 * score_continuidade)
    score_final = score_base - _penalidade_exposicao(geometria_bolsao.exposicao_superficial)
    return max(0.0, min(1.0, float(score_final)))


def avaliar_admissibilidade_bolsao(
    perfil_peptideo: PerfilPeptideo,
    geometria_bolsao: GeometriaBolsao,
) -> ResultadoAdmissibilidadeBolsao:
    """Avalia se um bolsão é geometricamente plausível para o peptídeo.

    Esta etapa é um filtro inicial de redução e não estima afinidade final.
    """

    motivos_reprovacao: list[str] = []

    comprimento_minimo = 0.80 * float(perfil_peptideo.comprimento_efetivo)
    largura_minima = 0.70 * float(perfil_peptideo.largura_efetiva)
    continuidade_minima = 0.25

    # Regra: comprimento útil precisa cobrir uma fração relevante do peptídeo.
    if geometria_bolsao.comprimento_util < comprimento_minimo:
        motivos_reprovacao.append("comprimento_util_insuficiente")

    # Regra: largura útil mínima evita regiões estreitas demais para acomodação.
    if geometria_bolsao.largura_util < largura_minima:
        motivos_reprovacao.append("largura_util_insuficiente")

    # Regra: continuidade baixa sugere geometria fragmentada pouco plausível.
    if geometria_bolsao.continuidade_superficial < continuidade_minima:
        motivos_reprovacao.append("continuidade_superficial_baixa")

    # Regra opcional: exposição extrema tende a reduzir plausibilidade geométrica.
    if geometria_bolsao.exposicao_superficial > 0.95:
        motivos_reprovacao.append("exposicao_superficial_excessiva")

    score = _calcular_score_encaixe(perfil_peptideo, geometria_bolsao)
    admissivel = len(motivos_reprovacao) == 0

    return ResultadoAdmissibilidadeBolsao(
        id_bolsao=geometria_bolsao.id_bolsao,
        admissivel=admissivel,
        score_encaixe_geometrico=score,
        motivos_reprovacao=tuple(motivos_reprovacao),
    )
