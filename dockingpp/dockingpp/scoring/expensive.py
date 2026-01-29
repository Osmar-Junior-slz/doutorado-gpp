"""Expensive scoring placeholder."""

from __future__ import annotations

from typing import Any, Optional

from dockingpp.data.structs import Pose


def _log_expensive_skip(cfg: Any, pose: Pose, reason: str) -> None:
    logger = getattr(cfg, "expensive_logger", None) or getattr(cfg, "logger", None)
    if logger is None:
        return
    step = int(getattr(cfg, "expensive_step", 0) or 0)
    extra = {"reason": reason}
    generation = pose.meta.get("generation")
    if generation is not None:
        extra["generation"] = generation
    logger.log_metric("expensive_skipped", 1.0, step=step, extra=extra)


def _score_pose_expensive_impl(pose: Pose, receptor: Any, peptide: Any, cfg: Any) -> float:
    """Compute an expensive score (placeholder)."""

    raise NotImplementedError("Integrar Vina/afins aqui.")


def score_pose_expensive(
    pose: Pose, receptor: Any, peptide: Any, cfg: Any
) -> Optional[float]:
    """Compute an expensive score (placeholder)."""

    try:
        return _score_pose_expensive_impl(pose, receptor, peptide, cfg)
    except NotImplementedError:
        _log_expensive_skip(cfg, pose, reason="not_implemented")
        return None
