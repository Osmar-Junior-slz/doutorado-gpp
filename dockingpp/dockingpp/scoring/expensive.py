"""Expensive scoring placeholder."""

from __future__ import annotations

from typing import Any

from dockingpp.data.structs import Pose


def score_pose_expensive(pose: Pose, receptor: Any, peptide: Any, cfg: Any) -> float:
    """Compute an expensive score (placeholder)."""

    raise NotImplementedError("Integrar Vina/afins aqui.")
