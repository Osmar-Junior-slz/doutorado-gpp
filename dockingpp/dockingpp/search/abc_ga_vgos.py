"""Minimal implementation of an ABC GA + VGOS-inspired search."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from dockingpp.data.structs import Population, Pose, RunResult
from dockingpp.pipeline.triggers import should_run_expensive
from dockingpp.search.engine import SearchEngine
from dockingpp.utils.geometry import apply_transform
from dockingpp.utils.topk import topk_indices


class ABCGAVGOSSearch(SearchEngine):
    """Minimal search engine with reproducible stochastic sampling."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def _get_base_coords(self, peptide: Any, cfg: Any, rng: np.random.Generator) -> np.ndarray:
        if isinstance(peptide, np.ndarray) and peptide.size:
            return np.asarray(peptide, dtype=float)
        num_atoms = int(getattr(cfg, "num_atoms", 0) or 0)
        num_atoms = num_atoms if num_atoms > 0 else 10
        return rng.normal(size=(num_atoms, 3)).astype(float)

    def _axis_angle_rotation(self, rng: np.random.Generator, max_rot_deg: float) -> np.ndarray:
        if max_rot_deg <= 0:
            return np.eye(3, dtype=float)
        angle = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.eye(3, dtype=float)
        axis = axis / norm
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=float,
        )

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
        base_coords = self._get_base_coords(peptide, cfg, rng)
        generations = int(getattr(cfg, "generations", 1) or 1)
        pop_size = int(getattr(cfg, "pop_size", getattr(cfg, "population_size", 20)) or 20)
        topk = max(1, int(getattr(cfg, "topk", 1) or 1))
        max_trans = float(getattr(cfg, "max_trans", 0.0) or 0.0)
        max_rot_deg = float(getattr(cfg, "max_rot_deg", 0.0) or 0.0)

        def random_pose() -> Pose:
            rot = self._axis_angle_rotation(rng, max_rot_deg)
            translation = rng.uniform(-max_trans, max_trans, size=3)
            coords = apply_transform(base_coords, rot, translation)
            return Pose(coords=coords)

        poses = [random_pose() for _ in range(pop_size)]
        best_pose = poses[0]
        best_score = float("-inf")
        best_generation = 0
        population = Population(poses=poses, generation=0)

        for generation in range(generations):
            if generation > 0:
                mutation_trans = max_trans * 0.2
                mutation_rot_deg = max_rot_deg * 0.2
                selected = [poses[idx] for idx in topk_indices(scores, topk, largest=True)]
                poses = []
                poses.extend(selected)
                while len(poses) < pop_size:
                    parent = selected[rng.integers(0, len(selected))]
                    rot = self._axis_angle_rotation(rng, mutation_rot_deg)
                    translation = rng.uniform(-mutation_trans, mutation_trans, size=3)
                    coords = apply_transform(parent.coords, rot, translation)
                    poses.append(Pose(coords=coords))

            scores = np.zeros(len(poses), dtype=float)
            n_clashes = 0.0
            for idx, pose in enumerate(poses):
                pose.score_cheap = score_cheap_fn(pose, pocket, cfg.cheap_weights)
                scores[idx] = pose.score_cheap or 0.0
                n_clashes += float(pose.meta.get("clashes", 0.0))

            population = Population(poses=poses, generation=generation)
            ranked = topk_indices(scores, topk, largest=True)
            for rank, idx in enumerate(ranked):
                if should_run_expensive(population.generation, rank, cfg):
                    poses[idx].score_expensive = score_expensive_fn(
                        poses[idx], receptor, peptide, cfg
                    )

            gen_best_idx = int(ranked[0]) if len(ranked) else int(np.argmax(scores))
            gen_best = poses[gen_best_idx]
            gen_best_score = gen_best.score_cheap or 0.0
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_pose = gen_best
                best_generation = generation

            logger.log_metric("best_score", float(gen_best_score), step=generation)
            logger.log_metric("mean_score", float(np.mean(scores)), step=generation)
            logger.log_metric("n_eval", float(len(poses)), step=generation)
            logger.log_metric("n_clashes", float(n_clashes), step=generation)

        best_pose.meta["generation"] = best_generation
        return RunResult(best_pose=best_pose, population=population, metrics={})
