"""Geometric KD-tree scan utilities for pocket feasibility."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.utils.kdtree import build_kdtree, query_radius


def build_receptor_kdtree(receptor_coords: np.ndarray) -> Any:
    """Build a KD-tree for receptor coordinates."""

    return build_kdtree(np.asarray(receptor_coords, dtype=float))


def _random_rotation_matrix(rng: np.random.Generator, max_deg: float | None = None) -> np.ndarray:
    max_rad = np.pi if max_deg is None else np.deg2rad(float(max_deg))
    angles = rng.uniform(-max_rad, max_rad, size=3)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def sample_peptide_placements(
    peptide_coords: np.ndarray,
    pocket_center: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    translation_sigma: float = 1.0,
    rotation_max_deg: float | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate random rigid-body transforms around a pocket center."""

    coords = np.asarray(peptide_coords, dtype=float)
    pocket_center = np.asarray(pocket_center, dtype=float)
    transforms: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(int(n_samples)):
        rotation = _random_rotation_matrix(rng, max_deg=rotation_max_deg)
        translation = pocket_center + rng.normal(scale=float(translation_sigma), size=3)
        transforms.append((rotation, translation))
    return transforms


def _rigid_transform(peptide_coords: np.ndarray, rotation: np.ndarray, new_center: np.ndarray) -> np.ndarray:
    coords = np.asarray(peptide_coords, dtype=float)
    if coords.size == 0:
        return coords.reshape(0, 3)
    center = coords.mean(axis=0)
    base_coords = coords - center
    return base_coords @ rotation.T + np.asarray(new_center, dtype=float)


def _receptor_points(receptor_kdtree: Any) -> np.ndarray:
    if isinstance(receptor_kdtree, dict):
        return np.asarray(receptor_kdtree.get("points", []), dtype=float)
    return np.asarray(getattr(receptor_kdtree, "data", []), dtype=float)


def _nearest_dists(receptor_kdtree: Any, points: np.ndarray, receptor_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0 or receptor_points.size == 0:
        return np.full((pts.shape[0],), np.inf, dtype=float)
    query = getattr(receptor_kdtree, "query", None)
    if callable(query):
        dists, _ = query(pts, k=1)
        return np.asarray(dists, dtype=float)
    # fallback path without scipy
    deltas = receptor_points[None, :, :] - pts[:, None, :]
    return np.linalg.norm(deltas, axis=2).min(axis=1)


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
    receptor_points = _receptor_points(receptor_kdtree)

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


def scan_pocket_feasibility(
    receptor_kdtree: Any,
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Legacy pocket scan based on clash/contact aggregates."""

    n_samples = int(_cfg_value(cfg_scan, "samples_per_pocket", 0) or 0)
    clash_cutoff = float(_cfg_value(cfg_scan, "clash_cutoff", 2.0) or 2.0)
    contact_cutoff = float(_cfg_value(cfg_scan, "contact_cutoff", 6.0) or 6.0)
    max_clash_ratio = float(_cfg_value(cfg_scan, "max_clash_ratio", 0.02) or 0.02)

    coords = np.asarray(peptide_coords, dtype=float)
    contacts_list: List[float] = []
    clashes_list: List[float] = []
    clash_ratios: List[float] = []

    transforms = sample_peptide_placements(coords, pocket.center, n_samples, rng)
    for rotation, translation in transforms:
        placed = _rigid_transform(coords, rotation, translation)
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


def _evaluate_geom_sample(
    receptor_kdtree: Any,
    receptor_points: np.ndarray,
    placed_coords: np.ndarray,
    pocket_center: np.ndarray,
    pocket_radius: float,
    cfg_scan: Any,
) -> dict[str, float]:
    severe_th = float(_cfg_value(cfg_scan, "severe_clash_threshold", 1.0))
    moderate_th = float(_cfg_value(cfg_scan, "moderate_clash_threshold", 1.1))
    contact_min = float(_cfg_value(cfg_scan, "contact_min", 1.5))
    contact_max = float(_cfg_value(cfg_scan, "contact_max", 3.0))
    exposure_margin = float(_cfg_value(cfg_scan, "exposure_margin", 2.0))

    dists = _nearest_dists(receptor_kdtree, placed_coords, receptor_points)
    if dists.size == 0:
        return {
            "dmin_global": float("inf"),
            "mean_d": float("inf"),
            "contacts": 0.0,
            "severe_clash": 1.0,
            "moderate_clash": 0.0,
            "pen_col": 10.0,
            "pen_exp": 10.0,
            "energy": 100.0,
        }

    severe_mask = dists < severe_th
    moderate_mask = (dists >= severe_th) & (dists < moderate_th)
    contact_mask = (dists >= contact_min) & (dists <= contact_max)

    severe_clash = float(np.any(severe_mask))
    moderate_clash = float(np.mean(moderate_mask))
    contacts = float(np.sum(contact_mask))

    severe_pen = 10.0 * float(np.sum((severe_th - dists[severe_mask]) ** 2))
    moderate_pen = float(np.sum((moderate_th - dists[moderate_mask]) ** 2))
    pen_col = severe_pen + moderate_pen

    centroid = placed_coords.mean(axis=0)
    center_delta = float(np.linalg.norm(centroid - np.asarray(pocket_center, dtype=float)))
    allowed = float(pocket_radius) + exposure_margin
    # penaliza afastamento do centro do bolsão acima da faixa do bolsão
    pen_exp = max(0.0, center_delta - allowed)

    # energia geométrica barata (menor = melhor)
    energy = pen_col + (0.25 * pen_exp) - (0.1 * contacts)

    return {
        "dmin_global": float(np.min(dists)),
        "mean_d": float(np.mean(dists)),
        "contacts": contacts,
        "severe_clash": severe_clash,
        "moderate_clash": moderate_clash,
        "pen_col": float(pen_col),
        "pen_exp": float(pen_exp),
        "energy": float(energy),
    }


def scan_pocket_feasibility_geom_kdtree(
    receptor_kdtree: Any,
    receptor_coords: np.ndarray,
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Enhanced rigid geometric scan using KD-tree distances."""

    n_samples = int(_cfg_value(cfg_scan, "samples_per_pocket", 64))
    translation_sigma = float(_cfg_value(cfg_scan, "translation_sigma", 2.0))
    rotation_max_deg = float(_cfg_value(cfg_scan, "rotation_max_deg", 30.0))
    reject_feasible_leq = float(_cfg_value(cfg_scan, "reject_if_feasible_fraction_leq", 0.0))

    receptor_points = np.asarray(receptor_coords, dtype=float)
    coords = np.asarray(peptide_coords, dtype=float)
    pocket_center = np.asarray(pocket.center, dtype=float)

    transforms = sample_peptide_placements(
        coords,
        pocket_center,
        n_samples,
        rng,
        translation_sigma=translation_sigma,
        rotation_max_deg=rotation_max_deg,
    )

    energies: list[float] = []
    severe_values: list[float] = []
    sample_is_feasible: list[bool] = []
    moderate_values: list[float] = []
    contact_values: list[float] = []
    exposure_values: list[float] = []
    best_meta: dict[str, Any] | None = None

    for sample_idx, (rotation, translation) in enumerate(transforms):
        placed = _rigid_transform(coords, rotation, translation)
        meta = _evaluate_geom_sample(receptor_kdtree, receptor_points, placed, pocket_center, float(pocket.radius), cfg_scan)
        meta["sample_idx"] = int(sample_idx)
        meta["translation"] = np.asarray(translation, dtype=float).tolist()
        meta["rotation_matrix"] = np.asarray(rotation, dtype=float).tolist()
        energies.append(float(meta["energy"]))
        severe_values.append(float(meta["severe_clash"]))
        # definição explícita de viabilidade por amostra no modo geom_kdtree:
        # amostra é viável se não houver colisão severa.
        sample_is_feasible.append(float(meta["severe_clash"]) == 0.0)
        moderate_values.append(float(meta["moderate_clash"]))
        contact_values.append(float(meta["contacts"]))
        exposure_values.append(float(meta["pen_exp"]))
        if best_meta is None or float(meta["energy"]) < float(best_meta["energy"]):
            best_meta = meta

    if not energies:
        feasible_fraction = 0.0
        severe_clash_fraction = 1.0
        moderate_clash_mean = 0.0
        mean_contacts = 0.0
        best_contact_count = 0.0
        best_geom_energy = 100.0
        mean_exposure_penalty = 10.0
        best_meta = {"reason": "no_samples"}
    else:
        feasible_fraction = float(np.mean(sample_is_feasible))
        severe_clash_fraction = float(np.mean(severe_values))
        moderate_clash_mean = float(np.mean(moderate_values))
        mean_contacts = float(np.mean(contact_values))
        best_contact_count = float(np.max(contact_values))
        best_geom_energy = float(np.min(energies))
        mean_exposure_penalty = float(np.mean(exposure_values))

    alpha = float(_cfg_value(cfg_scan, "score_alpha", 1.0))
    beta = float(_cfg_value(cfg_scan, "score_beta", 1.0))
    gamma = float(_cfg_value(cfg_scan, "score_gamma", 1.0))
    delta = float(_cfg_value(cfg_scan, "score_delta", 1.0))

    pocket_scan_score = (
        -best_geom_energy
        + (alpha * feasible_fraction)
        + (beta * mean_contacts)
        - (gamma * severe_clash_fraction)
        - (delta * mean_exposure_penalty)
    )

    # compatibilidade com caminhos existentes
    clash_ratio_best = float(best_meta.get("moderate_clash", 1.0)) if isinstance(best_meta, dict) else 1.0
    scan_score = pocket_scan_score
    return {
        "feasible_fraction": float(feasible_fraction),
        "severe_clash_fraction": float(severe_clash_fraction),
        "moderate_clash_mean": float(moderate_clash_mean),
        "mean_contacts": float(mean_contacts),
        "best_contact_count": float(best_contact_count),
        "best_geom_energy": float(best_geom_energy),
        "mean_exposure_penalty": float(mean_exposure_penalty),
        "pocket_scan_score": float(pocket_scan_score),
        "n_samples": int(n_samples),
        "best_sample_meta": best_meta or {},
        "selector_mode": "geom_kdtree",
        "selector_mode_used": "geom_kdtree",
        # No modo geom_kdtree, estes são os campos canônicos para auditoria/ranking:
        # - pocket_scan_score (score final para ranking de bolsões)
        # - severe_clash_fraction (fração de amostras com colisão severa)
        # - best_geom_energy (menor energia geométrica observada)
        "canonical_geom_fields": ["pocket_scan_score", "severe_clash_fraction", "best_geom_energy"],
        # Campos legados mantidos somente por compatibilidade de schema/consumo.
        "compatibility_fields": ["scan_score", "clash_ratio_best"],
        "scan_score": float(scan_score),
        "clash_ratio_best": float(clash_ratio_best),
        "reject_if_feasible_fraction_leq": float(reject_feasible_leq),
    }


def select_pockets_from_scan(
    pockets: List[Pocket],
    scan_table: Dict[str, Dict[str, float]],
    top_k: int | None,
    selector_mode: str = "legacy",
) -> List[Pocket]:
    """Select pockets based on scan score."""

    if top_k is None or int(top_k) <= 0 or int(top_k) >= len(pockets):
        return list(pockets)

    score_key = "pocket_scan_score" if selector_mode == "geom_kdtree" else "scan_score"
    scored: List[Tuple[float, float, int, Pocket]] = []
    for idx, pocket in enumerate(pockets):
        metrics = scan_table.get(str(pocket.id), {})
        score = float(metrics.get(score_key, float("-inf")))
        feasible = float(metrics.get("feasible_fraction", 0.0))
        scored.append((score, feasible, idx, pocket))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored[: int(top_k)]]
