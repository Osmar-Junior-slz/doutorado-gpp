"""Descrição geométrica simplificada de bolsões candidatos."""

from __future__ import annotations

from typing import Any

import numpy as np

from dockingpp.reducao.modelos import GeometriaBolsao


def _normalizar_coordenadas_bolsao(bolsao: Any) -> np.ndarray:
    """Normaliza coordenadas do bolsão para um `ndarray` Nx3.

    A função reutiliza o formato de dados já usado no projeto para bolsões
    (`Pocket` com atributo `coords`) e também aceita `meta["coords"]`.
    """

    coords_brutas = getattr(bolsao, "coords", None)
    if (coords_brutas is None or np.asarray(coords_brutas).size == 0) and hasattr(bolsao, "meta"):
        meta = getattr(bolsao, "meta", {}) or {}
        if isinstance(meta, dict):
            coords_brutas = meta.get("coords")

    if coords_brutas is None:
        return np.zeros((0, 3), dtype=float)

    coords = np.asarray(coords_brutas, dtype=float)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("As coordenadas do bolsão devem ter formato Nx3.")

    mascara_validos = np.all(np.isfinite(coords), axis=1)
    return coords[mascara_validos]


def _dimensoes_principais(coords_centralizadas: np.ndarray) -> tuple[float, float, float]:
    """Estima dimensões principais por projeção em eixos de PCA."""

    if coords_centralizadas.shape[0] <= 1:
        return 0.0, 0.0, 0.0

    cov = np.cov(coords_centralizadas, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(cov)
    ordem = np.argsort(autovalores)[::-1]
    eixos = autovetores[:, ordem]

    projecoes = coords_centralizadas @ eixos
    amplitudes = np.ptp(projecoes, axis=0)
    amplitudes_ordenadas = np.sort(np.asarray(amplitudes, dtype=float))[::-1]
    return (
        float(amplitudes_ordenadas[0]),
        float(amplitudes_ordenadas[1]),
        float(amplitudes_ordenadas[2]),
    )


def _continuidade_superficial(coords: np.ndarray, escala: float) -> float:
    """Calcula continuidade superficial por densidade de vizinhança local.

    Esta métrica é heurística: mede a fração média de vizinhos próximos para
    inferir se os pontos do bolsão formam uma região contínua.
    """

    n = coords.shape[0]
    if n <= 1:
        return 0.0

    limiar = max(1e-6, float(escala) * 0.35)
    deltas = coords[:, None, :] - coords[None, :, :]
    distancias = np.linalg.norm(deltas, axis=2)
    vizinhos = (distancias <= limiar).astype(float)
    np.fill_diagonal(vizinhos, 0.0)
    fracao_media = float(vizinhos.sum(axis=1).mean() / max(1.0, n - 1.0))
    return float(np.clip(fracao_media, 0.0, 1.0))


def _exposicao_superficial(coords: np.ndarray, centro: np.ndarray, raio: float) -> float:
    """Estima exposição superficial pela distância média ao centro do bolsão."""

    if coords.shape[0] == 0:
        # TODO: calibrar valor neutro de exposição quando faltarem coordenadas.
        return 0.5

    if raio <= 0.0:
        distancias = np.linalg.norm(coords - centro.reshape(1, 3), axis=1)
        raio = float(distancias.max()) if distancias.size else 0.0
        if raio <= 0.0:
            return 0.0

    dist_media = float(np.linalg.norm(coords - centro.reshape(1, 3), axis=1).mean())
    return float(np.clip(dist_media / raio, 0.0, 1.0))


def descrever_geometria_bolsao(bolsao: Any) -> GeometriaBolsao:
    """Gera uma descrição geométrica simples de um bolsão candidato.

    Espera um objeto compatível com `Pocket` do projeto atual, usando campos
    `id`, `center`, `radius` e `coords`.
    """

    id_bolsao = str(getattr(bolsao, "id", "desconhecido"))
    centro = np.asarray(getattr(bolsao, "center", np.zeros(3, dtype=float)), dtype=float)
    if centro.shape != (3,):
        centro = np.asarray(centro, dtype=float).reshape(-1)[:3]
        if centro.shape[0] < 3:
            centro = np.pad(centro, (0, 3 - centro.shape[0]))
    raio = float(getattr(bolsao, "radius", 0.0) or 0.0)

    coords = _normalizar_coordenadas_bolsao(bolsao)
    if coords.shape[0] == 0:
        diametro = max(0.0, 2.0 * raio)
        comprimento = diametro
        largura = diametro
        profundidade = diametro
        continuidade = 0.0
        exposicao = _exposicao_superficial(coords, centro, raio)
    else:
        centroide = coords.mean(axis=0)
        coords_centralizadas = coords - centroide.reshape(1, 3)
        comprimento, largura, profundidade = _dimensoes_principais(coords_centralizadas)
        continuidade = _continuidade_superficial(coords, escala=max(comprimento, largura, profundidade))
        exposicao = _exposicao_superficial(coords, centro, raio)

    # Aproximação por elipsoide para estimativa rápida do volume útil.
    volume_estimado = float((np.pi / 6.0) * comprimento * largura * profundidade)

    return GeometriaBolsao(
        id_bolsao=id_bolsao,
        comprimento_util=float(max(0.0, comprimento)),
        largura_util=float(max(0.0, largura)),
        profundidade_util=float(max(0.0, profundidade)),
        continuidade_superficial=continuidade,
        exposicao_superficial=exposicao,
        volume_estimado=float(max(0.0, volume_estimado)),
    )
