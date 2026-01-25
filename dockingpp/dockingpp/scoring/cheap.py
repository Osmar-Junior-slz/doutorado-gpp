"""Cheap scoring implementation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from dockingpp.data.structs import Pocket, Pose
from dockingpp.utils.geometry import pairwise_dist


def score_pose_cheap(pose: Pose, pocket: Pocket, weights: Dict[str, float]) -> float:
    """Compute a simple cheap score for a pose."""

    coords = pose.coords
    center = pocket.center.reshape(1, 3)
    dists = pairwise_dist(coords, center).flatten()

    contact_thresh = pocket.radius
    clash_thresh = max(0.5, pocket.radius * 0.3)

    contacts = float(np.sum(dists <= contact_thresh))
    clashes = float(np.sum(dists <= clash_thresh))
    geom_penalty = float(np.mean(dists))

    w_contacts = weights.get("w_contacts", 1.0)
    w_clashes = weights.get("w_clashes", 1.0)
    w_geom = weights.get("w_geom", 0.1)

    return w_contacts * contacts - w_clashes * clashes - w_geom * geom_penalty
