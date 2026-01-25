"""Trigger utilities for expensive scoring."""

from __future__ import annotations

from typing import List

from dockingpp.data.structs import Population


def should_run_expensive(gen: int, rank: int, cfg: object) -> bool:
    """Determine whether to run expensive scoring."""

    if getattr(cfg, "expensive_every", 0) <= 0:
        return False
    if gen % getattr(cfg, "expensive_every", 1) != 0:
        return False
    topk = getattr(cfg, "expensive_topk", None)
    if topk is not None and rank >= topk:
        return False
    return True


def select_for_expensive(population: Population, cfg: object) -> List[int]:
    """Select pose indices for expensive scoring."""

    topk = getattr(cfg, "expensive_topk", 0)
    return list(range(min(len(population.poses), max(0, topk))))
