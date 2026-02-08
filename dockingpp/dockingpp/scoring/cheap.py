"""Cheap scoring implementation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from dockingpp.data.structs import Pocket, Pose
from dockingpp.utils.kdtree import build_kdtree, query_radius


def score_pose_cheap(pose: Pose, pocket: Pocket, weights: Dict[str, float]) -> float:
    """Compute a heuristic geometric score (not a physical energy model)."""

    pose_coords = np.asarray(pose.coords, dtype=float)
    pocket_coords_source = "meta.coords"
    pocket_coords = pocket.meta.get("coords")
    if pocket_coords is None:
        pocket_coords_source = "meta.atoms"
        pocket_coords = pocket.meta.get("atoms")
    if pocket_coords is None and getattr(pocket, "coords", None) is not None:
        pocket_coords_source = "pocket.coords"
        pocket_coords = pocket.coords
    if pocket_coords is None:
        pocket_coords_source = "center"
        pocket_coords = pocket.center.reshape(1, 3)
    pocket_coords = np.asarray(pocket_coords, dtype=float)

    if pose_coords.size == 0 or pocket_coords.size == 0:
        debug_logger = pocket.meta.get("debug_logger") if isinstance(pocket.meta, dict) else None
        if debug_logger is not None:
            reason = "empty_pose_coords" if pose_coords.size == 0 else "empty_pocket_coords"
            debug_logger.log(
                {
                    "type": "cheap_score_zero",
                    "reason": reason,
                    "pose_coords_n": int(pose_coords.shape[0]),
                    "pocket_coords_n": int(pocket_coords.shape[0]),
                    "pocket_id": getattr(pocket, "id", None),
                    "pocket_coords_source": pocket_coords_source,
                }
            )
        return 0.0

    clash_thresh = 2.0
    contact_thresh = 6.0

    tree = build_kdtree(pocket_coords)
    neighbors = query_radius(tree, pose_coords, contact_thresh)

    contacts = 0.0
    clashes = 0.0
    for pose_idx, pocket_idxs in enumerate(neighbors):
        if pocket_idxs.size == 0:
            continue
        deltas = pocket_coords[pocket_idxs] - pose_coords[pose_idx]
        dists = np.linalg.norm(deltas, axis=1)
        clashes += float(np.sum(dists < clash_thresh))
        contacts += float(np.sum((dists >= clash_thresh) & (dists <= contact_thresh)))

    # Prefer singular keys when both singular/plural are provided.
    if "w_contact" in weights:
        w_contact = weights["w_contact"]
    elif "w_contacts" in weights:
        w_contact = weights["w_contacts"]
    else:
        w_contact = 1.0
    if "w_clash" in weights:
        w_clash = weights["w_clash"]
    elif "w_clashes" in weights:
        w_clash = weights["w_clashes"]
    else:
        w_clash = 1.0

    pose.meta["contacts"] = contacts
    pose.meta["clashes"] = clashes
    return w_contact * contacts - w_clash * clashes
