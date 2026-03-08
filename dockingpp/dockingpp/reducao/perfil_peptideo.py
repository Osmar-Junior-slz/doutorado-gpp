"""Construção de perfil geométrico simplificado para peptídeos."""

from __future__ import annotations

from typing import Any

import numpy as np

from dockingpp.reducao.modelos import PerfilPeptideo


def _normalizar_coordenadas_peptideo(peptideo: Any) -> np.ndarray:
    """Extrai e normaliza coordenadas do peptídeo para um `ndarray` Nx3.

    Aceita a representação já usada no projeto:
    - `dict` com chave `coords`;
    - `numpy.ndarray` direto;
    - objeto com atributo `coords`.
    """

    if isinstance(peptideo, dict) and "coords" in peptideo:
        coords_brutas = peptideo["coords"]
    elif isinstance(peptideo, np.ndarray):
        coords_brutas = peptideo
    else:
        coords_brutas = getattr(peptideo, "coords", None)

    if coords_brutas is None:
        return np.zeros((0, 3), dtype=float)

    coords = np.asarray(coords_brutas, dtype=float)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("As coordenadas do peptídeo devem ter formato Nx3.")

    mascara_validos = np.all(np.isfinite(coords), axis=1)
    return coords[mascara_validos]


def _extensao_maxima(coords: np.ndarray) -> float:
    """Calcula a maior distância entre dois pontos do conjunto."""

    if coords.shape[0] <= 1:
        return 0.0

    # TODO: para peptídeos muito grandes, trocar por busca aproximada em vez de O(n²).
    deltas = coords[:, None, :] - coords[None, :, :]
    distancias = np.linalg.norm(deltas, axis=2)
    return float(np.max(distancias))


def _dimensoes_principais(coords_centralizadas: np.ndarray) -> tuple[float, float, float]:
    """Estima comprimento, largura e espessura pelos eixos principais (PCA)."""

    if coords_centralizadas.shape[0] <= 1:
        return 0.0, 0.0, 0.0

    cov = np.cov(coords_centralizadas, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(cov)
    ordem = np.argsort(autovalores)[::-1]
    base_principal = autovetores[:, ordem]

    projecoes = coords_centralizadas @ base_principal
    amplitudes = np.ptp(projecoes, axis=0)
    amplitudes_ordenadas = np.sort(np.asarray(amplitudes, dtype=float))[::-1]

    comprimento = float(amplitudes_ordenadas[0])
    largura = float(amplitudes_ordenadas[1])
    espessura = float(amplitudes_ordenadas[2])
    return comprimento, largura, espessura


def _indice_flexibilidade(comprimento: float, largura: float, espessura: float) -> float:
    """Retorna um índice heurístico de flexibilidade estrutural.

    A heurística cresce quando há mais espalhamento em múltiplos eixos e é
    limitada ao intervalo [0, 1].
    """

    if comprimento <= 0.0:
        return 0.0

    # TODO: validar esta heurística com dados experimentais de flexibilidade.
    indice = (largura + espessura) / (2.0 * comprimento)
    return float(np.clip(indice, 0.0, 1.0))


def construir_perfil_peptideo(peptideo: Any) -> PerfilPeptideo:
    """Constrói um perfil geométrico simplificado do peptídeo.

    A entrada pode ser um `dict` com `coords`, um `numpy.ndarray` Nx3 ou um
    objeto com atributo `coords`, alinhado ao formato já utilizado no pipeline.
    """

    coords = _normalizar_coordenadas_peptideo(peptideo)
    if coords.shape[0] == 0:
        return PerfilPeptideo(
            comprimento_efetivo=0.0,
            largura_efetiva=0.0,
            espessura_efetiva=0.0,
            extensao_maxima=0.0,
            raio_giro=0.0,
            indice_flexibilidade=0.0,
        )

    centroide = coords.mean(axis=0)
    coords_centralizadas = coords - centroide.reshape(1, 3)

    comprimento, largura, espessura = _dimensoes_principais(coords_centralizadas)
    extensao = _extensao_maxima(coords)
    raio_giro = float(np.sqrt(np.mean(np.sum(coords_centralizadas**2, axis=1))))
    indice_flexibilidade = _indice_flexibilidade(comprimento, largura, espessura)

    return PerfilPeptideo(
        comprimento_efetivo=comprimento,
        largura_efetiva=largura,
        espessura_efetiva=espessura,
        extensao_maxima=extensao,
        raio_giro=raio_giro,
        indice_flexibilidade=indice_flexibilidade,
    )
