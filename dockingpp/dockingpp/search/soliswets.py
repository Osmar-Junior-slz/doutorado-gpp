"""Solis-Wets refinement stub."""

from __future__ import annotations

from dockingpp.data.structs import Pocket, Pose


def solis_wets_refine(pose: Pose, pocket: Pocket, cfg: object, score_fn: object) -> Pose:
    """Return a refined pose (stub)."""

    _ = (pocket, cfg, score_fn)
    return pose
