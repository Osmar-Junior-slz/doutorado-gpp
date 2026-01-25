"""Stub implementation of an ABC GA + VGOS search."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from dockingpp.data.structs import Population, Pose, RunResult
from dockingpp.pipeline.triggers import should_run_expensive
from dockingpp.utils.topk import topk_indices


class ABCGAVGOSSearch:
    """Placeholder search engine."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def search(
        self,
        receptor: Any,
        peptide: Any,
        pockets: list[Any],
        cfg: Any,
        score_cheap_fn: Callable[..., float],
        score_expensive_fn: Callable[..., float],
        prior_pocket: Any,
        prior_pose: Any,
        logger: Any,
    ) -> RunResult:
        """Run a minimal search loop."""

        _ = (prior_pocket, prior_pose)
        rng = np.random.default_rng(cfg.seed)
        pocket = pockets[0]
        poses: list[Pose] = []
        for _ in range(cfg.population_size):
            coords = rng.normal(size=(cfg.num_atoms, 3)).astype(float)
            pose = Pose(coords=coords)
            pose.score_cheap = score_cheap_fn(pose, pocket, cfg.cheap_weights)
            poses.append(pose)

        population = Population(poses=poses, generation=0)
        scores = np.array([pose.score_cheap or 0.0 for pose in poses])
        topk = topk_indices(scores, cfg.topk, largest=True)

        for rank, idx in enumerate(topk):
            if should_run_expensive(population.generation, rank, cfg):
                poses[idx].score_expensive = score_expensive_fn(
                    poses[idx], receptor, peptide, cfg
                )

        best_idx = int(topk[0]) if len(topk) else int(np.argmax(scores))
        best_pose = poses[best_idx]
        logger.log_metric("best_score_cheap", best_pose.score_cheap or 0.0, step=0)
        return RunResult(best_pose=best_pose, population=population, metrics={})
