"""Pipeline entrypoints."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field

from dockingpp.data.io import load_peptide, load_pockets, load_receptor
from dockingpp.data.structs import Pocket, RunResult
from dockingpp.pipeline.logging import RunLogger
from dockingpp.priors.pocket import PriorNetPocket, rank_pockets
from dockingpp.priors.pose import PriorNetPose
from dockingpp.reporting.models import RunSummary
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.scoring.expensive import score_pose_expensive
from dockingpp.search.abc_ga_vgos import ABCGAVGOSSearch
from dockingpp.utils.debug_logger import DebugLogger


class Config(BaseModel):
    seed: int = 7
    device: str = "cpu"
    generations: int = 5
    pop_size: int = 20
    topk: int = 5
    num_atoms: int = 10
    max_trans: float = 5.0
    max_rot_deg: float = 25.0
    cheap_weights: Dict[str, float] = Field(default_factory=dict)
    expensive_every: int = 0
    expensive_topk: Optional[int] = None
    top_pockets: int = 3
    full_search: bool = True
    max_pockets_used: int = 8
    search_space_mode: str = "global"
    debug_log_enabled: bool = False
    debug_log_path: Optional[str] = None
    debug_log_level: str = "INFO"

    class Config:
        extra = "allow"


def _dummy_inputs() -> tuple[Any, Any, list[Pocket]]:
    receptor_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    pockets = [Pocket(id="dummy-0", center=np.array([0.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords)]
    return {"coords": receptor_coords}, {"coords": receptor_coords}, pockets


def _extract_coords(receptor: Any) -> np.ndarray:
    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    return np.asarray(getattr(receptor, "coords", np.zeros((0, 3))), dtype=float)


def _build_global_pocket(receptor: Any, cfg: Config) -> Pocket:
    coords = _extract_coords(receptor)
    center = coords.mean(axis=0) if coords.size else np.zeros(3, dtype=float)
    radius = float(np.max(np.linalg.norm(coords - center, axis=1))) + float(getattr(cfg, "pocket_margin", 2.0)) if coords.size else 2.0
    return Pocket(id="global", center=center, radius=radius, coords=coords, meta={"coords": coords})


def _build_summary(
    run_id: str,
    cfg: Config,
    receptor_path: str,
    peptide_path: str,
    status: str,
    runtime_sec: float,
    total_pockets: int,
    selected_pockets: int,
    logger: RunLogger,
    result: RunResult | None,
    exc: Exception | None,
) -> RunSummary:
    expensive_count = sum(1 for r in logger.records if r.get("event") == "expensive_eval")
    trigger_count = sum(1 for r in logger.records if r.get("event") == "trigger_expensive")
    cheap_evals = sum(1 for r in logger.records if r.get("event") == "search_iteration")
    omega_full = float(total_pockets)
    omega_reduced = float(selected_pockets)
    return RunSummary(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        mode="single",
        engine="ABCGAVGOSSearch",
        receptor=receptor_path,
        peptide=peptide_path,
        runtime_sec=runtime_sec,
        omega_full=omega_full,
        omega_reduced=omega_reduced,
        omega_ratio=(omega_reduced / omega_full) if omega_full else 0.0,
        n_pockets_total=total_pockets,
        n_pockets_selected=selected_pockets,
        n_evals_cheap=cheap_evals,
        n_evals_expensive=expensive_count,
        best_score_cheap=result.best_pose.score_cheap if result else None,
        best_score_expensive=result.best_pose.score_expensive if result else None,
        confidence_final=None,
        trigger_count_expensive=trigger_count,
        error_type=type(exc).__name__ if exc else None,
        error_message=str(exc) if exc else None,
    )


def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    start_total = time.perf_counter()
    run_id = datetime.utcnow().isoformat() + "Z"
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg.seed)

    debug_logger = DebugLogger(
        enabled=bool(getattr(cfg, "debug_log_enabled", False)),
        path=getattr(cfg, "debug_log_path", None) or os.path.join(out_dir, "debug", "debug.jsonl"),
        level=str(getattr(cfg, "debug_log_level", "INFO")),
    )
    debug_logger.run_id = run_id
    cfg.debug_logger = debug_logger

    logger = RunLogger(run_id=run_id, out_dir=out_dir)
    cfg.expensive_logger = logger
    logger.emit_event("run_started", context={"mode": cfg.search_space_mode})

    result: RunResult | None = None
    exc: Exception | None = None
    total_pockets = 0
    selected_pockets = 0

    try:
        if receptor_path == "__dummy__" and peptide_path == "__dummy__":
            receptor, peptide, dummy_pockets = _dummy_inputs()
        else:
            receptor, peptide = load_receptor(receptor_path), load_peptide(peptide_path)
            dummy_pockets = []

        if cfg.search_space_mode == "global":
            pockets = [_build_global_pocket(receptor, cfg)]
        else:
            pockets = dummy_pockets or load_pockets(receptor, cfg=cfg, pockets_path=getattr(cfg, "pockets_path", None))
            ranked = rank_pockets(receptor, pockets, peptide=peptide, debug_logger=debug_logger)
            pockets = [p for p, _ in ranked] if ranked else pockets
            if not cfg.full_search:
                pockets = pockets[: int(getattr(cfg, "top_pockets", len(pockets)) or len(pockets))]
            else:
                pockets = pockets[: int(getattr(cfg, "max_pockets_used", len(pockets)) or len(pockets))]

        total_pockets, selected_pockets = len(dummy_pockets or pockets), len(pockets)
        for i, pocket in enumerate(pockets):
            logger.emit_event("pocket_ranked", step=i, pocket_id=str(getattr(pocket, "id", i)), pocket_index=i)
            logger.emit_event("pocket_selected", step=i, pocket_id=str(getattr(pocket, "id", i)), pocket_index=i)

        search = ABCGAVGOSSearch(cfg)
        result = search.search(
            receptor=receptor,
            peptide=peptide,
            pockets=pockets,
            cfg=cfg,
            score_cheap_fn=score_pose_cheap,
            score_expensive_fn=score_pose_expensive,
            prior_pocket=PriorNetPocket(),
            prior_pose=PriorNetPose(),
            logger=logger,
        )
        logger.emit_event("run_finished", value=result.best_pose.score_cheap)
        return result
    except Exception as e:
        exc = e
        logger.safe_log_error(e)
        raise
    finally:
        runtime = time.perf_counter() - start_total
        summary = _build_summary(
            run_id=run_id,
            cfg=cfg,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            status="failed" if exc else "success",
            runtime_sec=runtime,
            total_pockets=total_pockets,
            selected_pockets=selected_pockets,
            logger=logger,
            result=result,
            exc=exc,
        )
        logger.save_summary(summary)
        logger.save_manifest(status=summary.status)
        debug_logger.close()
