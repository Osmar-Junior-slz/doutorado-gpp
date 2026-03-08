"""Modelos de dados do subpipeline de redução de bolsões.

Este módulo define apenas contratos de entrada e saída para as etapas de
redução, sem lógica de processamento.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfilPeptideo:
    """Resumo geométrico do peptídeo usado na triagem de bolsões."""

    comprimento_efetivo: float
    largura_efetiva: float
    espessura_efetiva: float
    extensao_maxima: float
    raio_giro: float
    indice_flexibilidade: float


@dataclass(frozen=True)
class GeometriaBolsao:
    """Descritores geométricos e superficiais de um bolsão candidato."""

    id_bolsao: str
    comprimento_util: float
    largura_util: float
    profundidade_util: float
    continuidade_superficial: float
    exposicao_superficial: float
    volume_estimado: float


@dataclass(frozen=True)
class ResultadoAdmissibilidadeBolsao:
    """Resultado da etapa de admissibilidade geométrica do bolsão."""

    id_bolsao: str
    admissivel: bool
    score_encaixe_geometrico: float
    motivos_reprovacao: tuple[str, ...]


@dataclass(frozen=True)
class ResultadoPreAfinidadeBolsao:
    """Resultado preliminar de afinidade para bolsões admissíveis."""

    id_bolsao: str
    score_pre_afinidade: float
    score_contatos: float
    penalidade_clash: float
    score_ancoragem: float


@dataclass(frozen=True)
class EntradaRankingBolsao:
    """Contrato consolidado para etapa de ranking dos bolsões."""

    id_bolsao: str
    admissivel: bool
    score_encaixe_geometrico: float
    score_pre_afinidade: float
    score_final: float
