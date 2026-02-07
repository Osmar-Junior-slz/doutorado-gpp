"""Trigger utilities for expensive scoring."""

from __future__ import annotations

from typing import Any, List

from dockingpp.data.structs import Population


def _log_expensive_skip(cfg: Any, gen: int, rank: int, reason: str) -> None:
    logger = getattr(cfg, "expensive_logger", None) or getattr(cfg, "logger", None)
    if logger is None:
        return
    step = int(getattr(cfg, "expensive_step", gen) or gen)
    extra = {"reason": reason, "generation": gen, "rank": rank}
    logger.log_metric("expensive_skipped", 1.0, step=step, extra=extra)


def should_run_expensive(gen: int, rank: int, cfg: object) -> bool:
    """Determine whether to run expensive scoring."""

    if getattr(cfg, "expensive_every", 0) <= 0:
        _log_expensive_skip(cfg, gen, rank, reason="disabled")
        return False
    if gen % getattr(cfg, "expensive_every", 1) != 0:
        _log_expensive_skip(cfg, gen, rank, reason="generation_mismatch")
        return False
    topk = getattr(cfg, "expensive_topk", None)
    if topk is not None and rank >= topk:
        _log_expensive_skip(cfg, gen, rank, reason="rank_outside_topk")
        return False
    return True


def select_for_expensive(population: Population, cfg: object) -> List[int]:
    """Select pose indices for expensive scoring."""

    topk = getattr(cfg, "expensive_topk", 0)
    return list(range(min(len(population.poses), max(0, topk))))
