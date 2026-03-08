"""API pública do subpipeline de redução de bolsões.

Este pacote expõe contratos e funções utilitárias para a redução
condicionada ao peptídeo, sem acoplamento direto ao pipeline principal.
"""

from .admissibilidade import avaliar_admissibilidade_bolsao
from .geometria_bolsao import descrever_geometria_bolsao
from .modelos import (
    EntradaRankingBolsao,
    GeometriaBolsao,
    PerfilPeptideo,
    ResultadoAdmissibilidadeBolsao,
    ResultadoPreAfinidadeBolsao,
)
from .perfil_peptideo import construir_perfil_peptideo
from .pre_afinidade import estimar_pre_afinidade_bolsao
from .ranking_bolsoes import ranquear_bolsoes_candidatos
from .seletor_bolsoes import selecionar_bolsoes_para_busca

__all__ = [
    "PerfilPeptideo",
    "GeometriaBolsao",
    "ResultadoAdmissibilidadeBolsao",
    "ResultadoPreAfinidadeBolsao",
    "EntradaRankingBolsao",
    "construir_perfil_peptideo",
    "descrever_geometria_bolsao",
    "avaliar_admissibilidade_bolsao",
    "estimar_pre_afinidade_bolsao",
    "ranquear_bolsoes_candidatos",
    "selecionar_bolsoes_para_busca",
]
