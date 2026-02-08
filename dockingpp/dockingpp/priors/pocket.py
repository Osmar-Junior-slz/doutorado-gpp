"""Pocket prior model stub."""

from __future__ import annotations

import numpy as np

from dockingpp.data.structs import Pocket

DEFAULT_RANK_WEIGHTS = {
    "w_size": 1.0,
    "w_compact": 1.0,
    "w_depth": 0.5,
    "w_proximity": 0.5,
}


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


def _extract_coords(payload: object | None) -> np.ndarray:
    if payload is None:
        return np.zeros((0, 3), dtype=float)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=float)
    if isinstance(payload, dict):
        coords = payload.get("coords")
        if coords is not None:
            return np.asarray(coords, dtype=float)
    coords = getattr(payload, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _load_rank_weights(receptor: object, pockets: list[Pocket], peptide: object | None) -> dict[str, float]:
    candidates = [receptor, peptide]
    if pockets:
        candidates.append(pockets[0].meta)
    for payload in candidates:
        if payload is None:
            continue
        if isinstance(payload, dict):
            weights = payload.get("pocket_rank_weights")
            if isinstance(weights, dict):
                return weights
            cfg = payload.get("cfg")
            if isinstance(cfg, dict):
                weights = cfg.get("pocket_rank_weights")
                if isinstance(weights, dict):
                    return weights
        cfg = getattr(payload, "cfg", None)
        if cfg is not None:
            weights = getattr(cfg, "pocket_rank_weights", None)
            if isinstance(weights, dict):
                return weights
    return DEFAULT_RANK_WEIGHTS.copy()


def _pocket_coords(pocket: Pocket, receptor_coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(getattr(pocket, "coords", np.zeros((0, 3))), dtype=float)
    if coords.size:
        return coords
    if receptor_coords.size == 0:
        return coords
    center = np.asarray(pocket.center, dtype=float)
    radius = float(pocket.radius)
    distances = np.linalg.norm(receptor_coords - center.reshape(1, 3), axis=1)
    return receptor_coords[distances <= radius]


def rank_pockets(
    receptor: object,
    pockets: list[Pocket],
    peptide: object | None = None,
    debug_logger: object | None = None,
) -> list[tuple[Pocket, float]]:
    """Rank pockets by simple deterministic heuristics."""

    coords = _receptor_coords(receptor)
    peptide_coords = _extract_coords(peptide)
    weights = DEFAULT_RANK_WEIGHTS.copy()
    weights.update(_load_rank_weights(receptor, pockets, peptide))
    ranked: list[tuple[Pocket, float, str, int]] = []
    peptide_centroid = peptide_coords.mean(axis=0) if peptide_coords.size else None
    for idx, pocket in enumerate(pockets):
        center = np.asarray(pocket.center, dtype=float)
        pocket_coords = _pocket_coords(pocket, coords)
        distances = (
            np.linalg.norm(pocket_coords - center.reshape(1, 3), axis=1)
            if pocket_coords.size
            else np.zeros(0, dtype=float)
        )
        n_atoms = float(pocket_coords.shape[0])
        f_size = float(np.log1p(n_atoms))
        mean_dist = float(distances.mean()) if distances.size else 0.0
        std_dist = float(distances.std()) if distances.size else 0.0
        f_compact = -mean_dist
        f_depth = -std_dist
        f_proximity = 0.0
        proximity_dist = None
        if peptide_centroid is not None:
            proximity_dist = float(np.linalg.norm(center - peptide_centroid))
            f_proximity = -proximity_dist
        score = (
            weights.get("w_size", 0.0) * f_size
            + weights.get("w_compact", 0.0) * f_compact
            + weights.get("w_depth", 0.0) * f_depth
            + weights.get("w_proximity", 0.0) * f_proximity
        )
        if hasattr(pocket, "meta") and pocket.meta is not None:
            meta = pocket.meta
            meta.setdefault("rank_components", {})
            meta["rank_components"].update(
                {
                    "f_size": f_size,
                    "f_compact": f_compact,
                    "f_depth_proxy": f_depth,
                    "f_proximity": f_proximity,
                    "mean_distance_to_center": mean_dist,
                    "std_distance_to_center": std_dist,
                    "n_atoms_in_pocket": n_atoms,
                    "peptide_distance": proximity_dist,
                    "weights": {
                        "w_size": float(weights.get("w_size", 0.0)),
                        "w_compact": float(weights.get("w_compact", 0.0)),
                        "w_depth": float(weights.get("w_depth", 0.0)),
                        "w_proximity": float(weights.get("w_proximity", 0.0)),
                    },
                    "score": float(score),
                }
            )
        pocket_id = str(getattr(pocket, "id", ""))
        ranked.append((pocket, float(score), pocket_id, idx))
    ranked.sort(key=lambda item: (-item[1], item[2], item[3]))
    ranked_pairs = [(pocket, score) for pocket, score, _, _ in ranked]
    if debug_logger is not None:
        if not ranked_pairs:
            debug_logger.log({"type": "pocket_fallback", "reason": "no_ranked", "n_in": int(len(pockets))})
        else:
            top_n = min(5, len(ranked_pairs))
            top_payload = []
            for pocket, score in ranked_pairs[:top_n]:
                meta = getattr(pocket, "meta", {}) or {}
                top_payload.append(
                    {
                        "pocket_id": str(getattr(pocket, "id", "")),
                        "score": float(score),
                        "rank_components": meta.get("rank_components", {}),
                    }
                )
            debug_logger.log(
                {
                    "type": "pocket_rank_summary",
                    "n_in": int(len(pockets)),
                    "n_out": int(len(ranked_pairs)),
                    "top": top_payload,
                    "weights": {key: float(value) for key, value in weights.items()},
                }
            )
    return ranked_pairs


class PriorNetPocket:
    """Stub prior network for pocket scoring."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return pocket prior scores."""

        return np.zeros(features.shape[0], dtype=float)
