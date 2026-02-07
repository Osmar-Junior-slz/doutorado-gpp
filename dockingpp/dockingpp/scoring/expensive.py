"""Expensive scoring placeholder."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dockingpp.data.structs import Pose
from dockingpp.utils.kdtree import build_kdtree, query_radius


def _get_expensive_cfg(cfg: Any) -> dict[str, float]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg.get("expensive", {}) if isinstance(cfg.get("expensive"), dict) else {}
    expensive_cfg = getattr(cfg, "expensive", None)
    if isinstance(expensive_cfg, dict):
        return expensive_cfg
    return {}


def _get_expensive_param(cfg: Any, key: str, default: float) -> float:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return float(cfg.get(key, default))
    return float(getattr(cfg, key, default))


def _extract_coords(source: Any) -> np.ndarray:
    if source is None:
        return np.zeros((0, 3), dtype=float)
    if isinstance(source, dict):
        coords = source.get("coords")
        if coords is None:
            coords = source.get("atoms")
        if coords is None:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(coords, dtype=float)
    if isinstance(source, np.ndarray):
        return np.asarray(source, dtype=float)
    coords = getattr(source, "coords", None)
    if coords is None:
        coords = getattr(source, "atoms", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _log_expensive_event(cfg: Any, name: str, pose: Pose, reason: str | None = None) -> None:
    logger = getattr(cfg, "expensive_logger", None) or getattr(cfg, "logger", None)
    if logger is None:
        return
    step = int(getattr(cfg, "expensive_step", 0) or 0)
    extra = {}
    if reason is not None:
        extra["reason"] = reason
    generation = pose.meta.get("generation")
    if generation is not None:
        extra["generation"] = generation
    if extra:
        logger.log_metric(name, 1.0, step=step, extra=extra)
    else:
        logger.log_metric(name, 1.0, step=step)


def _log_expensive_skip(cfg: Any, pose: Pose, reason: str) -> None:
    _log_expensive_event(cfg, "expensive_skipped", pose, reason=reason)


def _score_pose_expensive_impl(pose: Pose, receptor: Any, peptide: Any, cfg: Any) -> float:
    """Compute an expensive score (placeholder)."""

    pose_coords = np.asarray(pose.coords, dtype=float)
    receptor_coords = _extract_coords(receptor)
    if receptor_coords.size == 0:
        receptor_coords = _extract_coords(peptide)

    if pose_coords.size == 0 or receptor_coords.size == 0:
        pose.meta["expensive_contacts"] = 0.0
        pose.meta["expensive_clashes"] = 0.0
        return 0.0

    expensive_cfg = _get_expensive_cfg(cfg)
    contact_cutoff = _get_expensive_param(expensive_cfg, "contact_cutoff", 4.0)
    clash_cutoff = _get_expensive_param(expensive_cfg, "clash_cutoff", 2.0)
    w_att = _get_expensive_param(expensive_cfg, "w_att", 1.0)
    w_rep = _get_expensive_param(expensive_cfg, "w_rep", 3.0)

    if contact_cutoff <= 0:
        contact_cutoff = 4.0
    if clash_cutoff <= 0:
        clash_cutoff = 2.0

    tree = build_kdtree(receptor_coords)
    neighbors = query_radius(tree, pose_coords, contact_cutoff)

    contacts = 0.0
    clashes = 0.0
    for pose_idx, receptor_idxs in enumerate(neighbors):
        if receptor_idxs.size == 0:
            continue
        deltas = receptor_coords[receptor_idxs] - pose_coords[pose_idx]
        dists = np.linalg.norm(deltas, axis=1)
        contacts += float(np.sum(dists < contact_cutoff))
        clashes += float(np.sum(dists < clash_cutoff))

    pose.meta["expensive_contacts"] = contacts
    pose.meta["expensive_clashes"] = clashes

    return (w_rep * clashes) - (w_att * contacts)


def score_pose_expensive(
    pose: Pose, receptor: Any, peptide: Any, cfg: Any
) -> Optional[float]:
    """Compute an expensive score (placeholder)."""

    try:
        score = _score_pose_expensive_impl(pose, receptor, peptide, cfg)
    except NotImplementedError:
        _log_expensive_skip(cfg, pose, reason="not_implemented")
        return None
    _log_expensive_event(cfg, "expensive_ran", pose)
    return score
