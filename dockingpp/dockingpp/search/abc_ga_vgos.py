"""Minimal implementation of an ABC GA + VGOS-inspired search.

Nota (PT-BR): o comportamento anterior alternava pockets a cada geração e
misturava a população entre eles. Isso introduzia um "sawtooth" artificial
na convergência e mascarava a diferença entre modo full e reduced. A correção
abaixo executa o otimizador por pocket (sequencialmente), mantendo estado
isolado por pocket e tornando as métricas interpretáveis.
"""

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

    def _search_single_pocket(
        self,
        pocket: Any,
        rng: np.random.Generator,
        base_coords: np.ndarray,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        score_cheap_fn: Callable[..., float],
        score_expensive_fn: Callable[..., float],
        logger: Any,
        pocket_index: int,
        step_offset: int,
    ) -> tuple[Pose, Population]:
        """Executa a busca dentro de um único pocket.

        PT-BR: esta função garante que cada pocket tenha sua própria população
        e RNG, evitando a alternância não controlada que causava o sawtooth.
        Além disso, registramos a "generation" local (0..N) separada do "step"
        global, pois a UI deve exibir progresso por geração sem extrapolar N.
        """

        generations = int(getattr(cfg, "generations", 1) or 1)
        pop_size = int(getattr(cfg, "pop_size", getattr(cfg, "population_size", 20)) or 20)
        topk = max(1, int(getattr(cfg, "topk", 1) or 1))
        max_trans = float(getattr(cfg, "max_trans", 0.0) or 0.0)
        max_rot_deg = float(getattr(cfg, "max_rot_deg", 0.0) or 0.0)
        debug_logger = getattr(cfg, "debug_logger", None)

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

        if debug_logger is not None:
            debug_logger.log(
                {
                    "type": "search_pocket_start",
                    "pocket_id": getattr(pocket, "id", None),
                    "pocket_index": int(pocket_index),
                    "generations": int(generations),
                    "pop_size": int(pop_size),
                    "topk": int(topk),
                }
            )

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
            n_score_zero = 0
            for idx, pose in enumerate(poses):
                pose.score_cheap = score_cheap_fn(pose, pocket, cfg.cheap_weights)
                scores[idx] = pose.score_cheap or 0.0
                if scores[idx] == 0.0:
                    n_score_zero += 1
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
                best_generation = step_offset + generation
                best_pose.meta["pocket_id"] = getattr(pocket, "id", None)

            step = step_offset + generation
            # PT-BR: "step" é global (inclui offset de pocket) para séries; já
            # "generation" é local e limita o progresso a cfg.generations.
            extra = {
                "generation": generation,
                "total_generations": generations,
                "pocket_index": pocket_index,
            }
            logger.log_metric("best_score", float(gen_best_score), step=step, extra=extra)
            logger.log_metric("mean_score", float(np.mean(scores)), step=step, extra=extra)
            logger.log_metric("n_eval", float(len(poses)), step=step, extra=extra)
            logger.log_metric("n_clashes", float(n_clashes), step=step, extra=extra)
            if debug_logger is not None:
                debug_logger.log(
                    {
                        "type": "search_generation",
                        "pocket_id": getattr(pocket, "id", None),
                        "pocket_index": int(pocket_index),
                        "generation": int(generation),
                        "n_eval": int(len(poses)),
                        "best_score": float(gen_best_score),
                        "mean_score": float(np.mean(scores)),
                        "min_score": float(np.min(scores)) if scores.size else 0.0,
                        "max_score": float(np.max(scores)) if scores.size else 0.0,
                        "n_score_zero": int(n_score_zero),
                    }
                )

        best_pose.meta["generation"] = best_generation
        if debug_logger is not None:
            debug_logger.log(
                {
                    "type": "search_pocket_end",
                    "pocket_id": getattr(pocket, "id", None),
                    "best_score": float(best_score),
                    "best_pose_id": best_pose.meta.get("pose_id") or best_pose.meta.get("id"),
                }
            )
        return best_pose, population

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
        if not pockets:
            raise ValueError("No pockets available for search.")
        seed_seq = np.random.SeedSequence(cfg.seed)
        child_seqs = seed_seq.spawn(len(pockets) + 1)
        base_rng = np.random.default_rng(child_seqs[0])
        base_coords = self._get_base_coords(peptide, cfg, base_rng)

        # PT-BR: estratégia escolhida = otimizar por pocket de forma sequencial.
        # Isso evita o sawtooth (alternância de pocket sem estado isolado) e
        # torna a redução (full vs reduced) proporcional ao número de pockets.
        best_pose: Pose | None = None
        best_score = float("-inf")
        last_population: Population | None = None
        generations = int(getattr(cfg, "generations", 1) or 1)

        for pocket_idx, pocket in enumerate(pockets):
            rng = np.random.default_rng(child_seqs[pocket_idx + 1])
            step_offset = pocket_idx * generations
            pocket_best, population = self._search_single_pocket(
                pocket=pocket,
                rng=rng,
                base_coords=base_coords,
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                score_cheap_fn=score_cheap_fn,
                score_expensive_fn=score_expensive_fn,
                logger=logger,
                pocket_index=pocket_idx,
                step_offset=step_offset,
            )
            last_population = population
            pocket_score = pocket_best.score_cheap or 0.0
            if pocket_score > best_score or best_pose is None:
                best_score = pocket_score
                best_pose = pocket_best

        if best_pose is None or last_population is None:
            raise ValueError("No pockets available for search.")
        best_pose.meta.setdefault("generation", 0)
        return RunResult(best_pose=best_pose, population=last_population, metrics={})
