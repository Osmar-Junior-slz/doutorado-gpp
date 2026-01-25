"""Pocket prior model stub."""

from __future__ import annotations

import numpy as np

from dockingpp.data.structs import Pocket


def _receptor_coords(receptor: object) -> np.ndarray:
    if isinstance(receptor, np.ndarray):
        return receptor
    coords = None
    if isinstance(receptor, dict):
        coords = receptor.get("coords")
    if coords is None and hasattr(receptor, "coords"):
        coords = getattr(receptor, "coords")
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def rank_pockets(receptor: object, pockets: list[Pocket], peptide: object | None = None) -> list[tuple[Pocket, float]]:
    """Rank pockets by simple deterministic heuristics."""

    _ = peptide
    coords = _receptor_coords(receptor)
    ranked: list[tuple[Pocket, float, int]] = []
    margin = 2.0
    for idx, pocket in enumerate(pockets):
        center = np.asarray(pocket.center, dtype=float)
        radius = float(pocket.radius)
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
        ranked.append((pocket, float(score), idx))
    ranked.sort(key=lambda item: (-item[1], item[2]))
    return [(pocket, score) for pocket, score, _ in ranked]


class PriorNetPocket:
    """Stub prior network for pocket scoring."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return pocket prior scores."""

        return np.zeros(features.shape[0], dtype=float)
