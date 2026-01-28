"""Detecção de bolsões a partir do escaneamento do receptor."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from dockingpp.data.structs import Pocket


def _get_cfg_value(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _clusterizar_pontos(grid_indices: np.ndarray) -> list[list[int]]:
    """Agrupa índices em clusters com base em vizinhança 6-conexa.

    PT-BR: usamos grid indices inteiros e BFS simples para formar clusters de
    espaço vazio, mantendo implementação leve e reprodutível.
    """

    if grid_indices.size == 0:
        return []

    index_map: dict[tuple[int, int, int], int] = {
        (int(row[0]), int(row[1]), int(row[2])): idx for idx, row in enumerate(grid_indices)
    }
    visited = set()
    clusters: list[list[int]] = []
    neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for key, idx in index_map.items():
        if idx in visited:
            continue
        stack = [key]
        cluster = []
        while stack:
            current = stack.pop()
            current_idx = index_map.get(current)
            if current_idx is None or current_idx in visited:
                continue
            visited.add(current_idx)
            cluster.append(current_idx)
            for dx, dy, dz in neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if neighbor in index_map and index_map[neighbor] not in visited:
                    stack.append(neighbor)
        clusters.append(cluster)
    return clusters


def _iter_chunks(items: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    """Itera por chunks para evitar picos de memória em distâncias."""

    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def detectar_bolsoes(scan: Dict[str, Any], cfg: Any | None = None) -> List[Pocket]:
    """Detecta bolsões candidatos com base no escaneamento do receptor.

    PT-BR: implementa detecção baseada em geometria (grid de espaço vazio).
    Identificamos pontos do grid com distância ao receptor entre limites e
    clusterizamos esses pontos para definir bolsões candidatos.
    """

    coords = np.asarray(scan.get("coords", np.zeros((0, 3), dtype=float)), dtype=float)
    if coords.size == 0:
        return []

    grid_spacing = float(_get_cfg_value(cfg, "pocket_grid_spacing", 2.0))
    pocket_margin = float(_get_cfg_value(cfg, "pocket_margin", 2.0))
    min_dist = float(_get_cfg_value(cfg, "pocket_min_dist", 2.0))
    max_dist = float(_get_cfg_value(cfg, "pocket_max_dist", 6.0))
    min_cluster_size = int(_get_cfg_value(cfg, "pocket_min_cluster_points", 8) or 0)
    chunk_size = int(_get_cfg_value(cfg, "pocket_distance_chunk", 2000) or 2000)

    bbox_min = coords.min(axis=0) - max_dist
    bbox_max = coords.max(axis=0) + max_dist

    axes = [
        np.arange(bbox_min[i], bbox_max[i] + grid_spacing, grid_spacing) for i in range(3)
    ]
    grid_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    candidate_points: list[np.ndarray] = []
    candidate_indices: list[np.ndarray] = []

    for chunk in _iter_chunks(grid_points, chunk_size):
        deltas = chunk[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(deltas, axis=2)
        min_distances = dists.min(axis=1)
        mask = (min_distances >= min_dist) & (min_distances <= max_dist)
        if not np.any(mask):
            continue
        candidate_points.append(chunk[mask])
        candidate_indices.append(np.round((chunk[mask] - bbox_min) / grid_spacing).astype(int))

    if not candidate_points:
        return []

    points = np.vstack(candidate_points)
    grid_indices = np.vstack(candidate_indices)
    clusters = _clusterizar_pontos(grid_indices)

    pockets: list[Pocket] = []
    for idx, cluster in enumerate(clusters):
        if min_cluster_size and len(cluster) < min_cluster_size:
            continue
        cluster_points = points[cluster]
        center = cluster_points.mean(axis=0)
        deltas = cluster_points - center
        radius = float(np.max(np.linalg.norm(deltas, axis=1))) + pocket_margin
        deltas_receptor = coords - center.reshape(1, 3)
        mask = np.linalg.norm(deltas_receptor, axis=1) <= radius
        pocket_coords = coords[mask]
        pockets.append(
            Pocket(
                id=f"scan_cluster_{idx}",
                center=center,
                radius=radius,
                coords=pocket_coords,
                meta={"coords": pocket_coords, "grid_points": cluster_points},
            )
        )

    return pockets


def construir_bolso_global(coords: np.ndarray, cfg: Any | None = None) -> Pocket:
    """Constrói um bolso global como fallback explícito.

    PT-BR: só deve ser usado quando a detecção falha ou retorna zero bolsões.
    """

    pocket_margin = float(_get_cfg_value(cfg, "pocket_margin", 2.0))
    if coords.size:
        center = coords.mean(axis=0)
        deltas = coords - center.reshape(1, 3)
        max_dist = float(np.max(np.linalg.norm(deltas, axis=1)))
    else:
        center = np.zeros(3, dtype=float)
        max_dist = 0.0
    radius = max_dist + pocket_margin
    return Pocket(
        id="global_fallback",
        center=center,
        radius=radius,
        coords=coords,
        meta={"coords": coords, "fallback": True},
    )
