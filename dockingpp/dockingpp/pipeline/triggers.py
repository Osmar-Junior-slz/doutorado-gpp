"""Trigger utilities for expensive scoring."""

from __future__ import annotations

from typing import Any, List

from dockingpp.data.structs import Population


def _emit_trigger(cfg: Any, gen: int, rank: int, reason: str, threshold: float | None = None, pose_id: str | None = None) -> None:
    logger = getattr(cfg, "expensive_logger", None) or getattr(cfg, "logger", None)
    if logger is None:
        return
    step = int(getattr(cfg, "expensive_step", gen) or gen)
    logger.emit_event(
        "trigger_expensive",
        step=step,
        reason=reason,
        threshold=threshold,
        pose_id=pose_id,
        context={"generation": gen, "rank": rank, "decision": reason},
    )


def should_run_expensive(gen: int, rank: int, cfg: object) -> bool:
    if getattr(cfg, "expensive_every", 0) <= 0:
        _emit_trigger(cfg, gen, rank, reason="disabled", threshold=float(getattr(cfg, "expensive_every", 0) or 0))
        return False
    if gen % getattr(cfg, "expensive_every", 1) != 0:
        _emit_trigger(cfg, gen, rank, reason="generation_mismatch", threshold=float(getattr(cfg, "expensive_every", 1)))
        return False
    topk = getattr(cfg, "expensive_topk", None)
    if topk is not None and rank >= topk:
        _emit_trigger(cfg, gen, rank, reason="rank_outside_topk", threshold=float(topk))
        return False
    _emit_trigger(cfg, gen, rank, reason="accepted", threshold=float(topk) if topk is not None else None)
    return True


def select_for_expensive(population: Population, cfg: object) -> List[int]:
    topk = getattr(cfg, "expensive_topk", 0)
    return list(range(min(len(population.poses), max(0, topk))))
