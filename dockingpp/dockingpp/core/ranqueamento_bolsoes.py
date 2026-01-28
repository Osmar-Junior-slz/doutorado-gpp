"""Ranqueamento e seleção de bolsões candidatos (PT-BR)."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from dockingpp.data.structs import Pocket


def _receptor_coords(receptor: Any) -> np.ndarray:
    """Extrai coordenadas do receptor para uso em heurísticas de ranking."""

    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = None
    if isinstance(receptor, dict):
        coords = receptor.get("coords")
    if coords is None and hasattr(receptor, "coords"):
        coords = getattr(receptor, "coords")
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def ranquear_bolsoes(
    receptor: Any,
    bolsos: list[Pocket],
    peptide: Any | None = None,
) -> List[Tuple[Pocket, float]]:
    """Ranqueia bolsões via heurísticas simples e determinísticas.

    PT-BR: usa densidade de átomos no raio e contato com margem para priorizar
    bolsões mais promissores. Mantém determinismo por índice original.
    """

    _ = peptide
    coords = _receptor_coords(receptor)
    ranked: list[tuple[Pocket, float, int]] = []
    margin = 2.0
    for idx, bolso in enumerate(bolsos):
        center = np.asarray(bolso.center, dtype=float)
        radius = float(bolso.radius)
        if coords.size:
            deltas = coords - center
            distances = np.linalg.norm(deltas, axis=1)
            n_inside = float(np.sum(distances <= radius))
            n_contacts = float(np.sum(distances <= radius + margin))
        else:
            n_inside = 0.0
            n_contacts = 0.0
        volume = (4.0 / 3.0) * np.pi * radius**3 if radius > 0 else 0.0
        density = n_inside / volume if volume > 0 else 0.0
        score = density + n_contacts
        ranked.append((bolso, float(score), idx))
    ranked.sort(key=lambda item: (-item[1], item[2]))
    return [(bolso, score) for bolso, score, _ in ranked]


def selecionar_top_bolsoes(
    bolsos_rankeados: list[tuple[Pocket, float]],
    top_k: int | None,
    full_search: bool,
) -> list[Pocket]:
    """Seleciona bolsões a partir do ranking.

    PT-BR: se full_search=True, retorna todos; caso contrário aplica top-k.
    """

    if full_search:
        return [bolso for bolso, _ in bolsos_rankeados]
    if top_k is None or top_k <= 0:
        return []
    return [bolso for bolso, _ in bolsos_rankeados[:top_k]]


def rank_pockets(
    receptor: Any,
    pockets: list[Pocket],
    peptide: Any | None = None,
) -> List[Tuple[Pocket, float]]:
    """Alias retrocompatível em inglês para ranquear bolsões."""

    return ranquear_bolsoes(receptor, pockets, peptide=peptide)
