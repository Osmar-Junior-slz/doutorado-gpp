"""Geometric KD-tree scan utilities for pocket feasibility."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.utils.kdtree import build_kdtree, query_radius


def build_receptor_kdtree(receptor_coords: np.ndarray) -> Any:
    """Build a KD-tree for receptor coordinates."""

    return build_kdtree(np.asarray(receptor_coords, dtype=float))


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=3)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def sample_peptide_placements(
    peptide_coords: np.ndarray,
    pocket_center: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate random rotations/translations for a peptide around a pocket center."""

    coords = np.asarray(peptide_coords, dtype=float)
    center = coords.mean(axis=0) if coords.size else np.zeros(3, dtype=float)
    pocket_center = np.asarray(pocket_center, dtype=float)
    transforms: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(int(n_samples)):
        rotation = _random_rotation_matrix(rng)
        translation = pocket_center + rng.normal(scale=1.0, size=3)
        transforms.append((rotation, translation))
    return transforms


def feasibility_metrics(
    receptor_kdtree: Any,
    placed_peptide_coords: np.ndarray,
    clash_cutoff: float,
    contact_cutoff: float,
) -> tuple[float, float]:
    """Compute clash/contact counts for placed peptide coordinates."""

    coords = np.asarray(placed_peptide_coords, dtype=float)
    if coords.size == 0:
        return 0.0, 0.0
    neighbors = query_radius(receptor_kdtree, coords, float(contact_cutoff))
    if isinstance(receptor_kdtree, dict):
        receptor_points = np.asarray(receptor_kdtree.get("points", []), dtype=float)
    else:
        receptor_points = np.asarray(getattr(receptor_kdtree, "data", []), dtype=float)

    clashes = 0.0
    contacts = 0.0
    for idx, receptor_idxs in enumerate(neighbors):
        if receptor_idxs.size == 0:
            continue
        deltas = receptor_points[receptor_idxs] - coords[idx]
        dists = np.linalg.norm(deltas, axis=1)
        clashes += float(np.sum(dists < clash_cutoff))
        contacts += float(np.sum((dists >= clash_cutoff) & (dists <= contact_cutoff)))
    return clashes, contacts


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def scan_pocket_feasibility(
    receptor_kdtree: Any,
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Scan a pocket and aggregate feasibility metrics."""

    n_samples = int(_cfg_value(cfg_scan, "samples_per_pocket", 0) or 0)
    clash_cutoff = float(_cfg_value(cfg_scan, "clash_cutoff", 2.0) or 2.0)
    contact_cutoff = float(_cfg_value(cfg_scan, "contact_cutoff", 6.0) or 6.0)
    max_clash_ratio = float(_cfg_value(cfg_scan, "max_clash_ratio", 0.02) or 0.02)

    coords = np.asarray(peptide_coords, dtype=float)
    center = coords.mean(axis=0) if coords.size else np.zeros(3, dtype=float)
    base_coords = coords - center

    contacts_list: List[float] = []
    clashes_list: List[float] = []
    clash_ratios: List[float] = []

    transforms = sample_peptide_placements(coords, pocket.center, n_samples, rng)
    for rotation, translation in transforms:
        placed = base_coords @ rotation.T + translation
        clashes, contacts = feasibility_metrics(
            receptor_kdtree,
            placed,
            clash_cutoff=clash_cutoff,
            contact_cutoff=contact_cutoff,
        )
        contacts_list.append(float(contacts))
        clashes_list.append(float(clashes))
        ratio = float(clashes) / max(1.0, float(coords.shape[0]))
        clash_ratios.append(ratio)

    if not contacts_list:
        best_contacts = 0.0
        mean_contacts = 0.0
        best_clashes = 0.0
        clash_ratio_best = 0.0
        feasible_fraction = 0.0
        mean_clashes = 0.0
    else:
        best_contacts = float(np.max(contacts_list))
        mean_contacts = float(np.mean(contacts_list))
        best_clashes = float(np.min(clashes_list))
        mean_clashes = float(np.mean(clashes_list))
        clash_ratio_best = float(np.min(clash_ratios))
        feasible_fraction = float(np.mean([ratio <= max_clash_ratio for ratio in clash_ratios]))

    penalty = 1.0
    scan_score = feasible_fraction * mean_contacts - penalty * mean_clashes

    return {
        "best_contacts": best_contacts,
        "mean_contacts": mean_contacts,
        "best_clashes": best_clashes,
        "clash_ratio_best": clash_ratio_best,
        "feasible_fraction": feasible_fraction,
        "scan_score": scan_score,
    }


def select_pockets_from_scan(
    pockets: List[Pocket],
    scan_table: Dict[str, Dict[str, float]],
    top_k: int | None,
) -> List[Pocket]:
    """Select pockets based on scan score."""

    if top_k is None or int(top_k) <= 0 or int(top_k) >= len(pockets):
        return list(pockets)

    scored: List[Tuple[float, float, int, Pocket]] = []
    for idx, pocket in enumerate(pockets):
        metrics = scan_table.get(str(pocket.id), {})
        score = float(metrics.get("scan_score", float("-inf")))
        feasible = float(metrics.get("feasible_fraction", 0.0))
        scored.append((score, feasible, idx, pocket))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored[: int(top_k)]]
