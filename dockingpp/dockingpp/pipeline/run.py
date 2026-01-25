"""Pipeline entrypoints."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, Field

from dockingpp.data.io import load_peptide, load_pockets, load_receptor
from dockingpp.data.structs import Pocket, RunResult
from dockingpp.pipeline.logging import RunLogger
from dockingpp.priors.pocket import PriorNetPocket
from dockingpp.priors.pose import PriorNetPose
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.scoring.expensive import score_pose_expensive
from dockingpp.search.abc_ga_vgos import ABCGAVGOSSearch


class Config(BaseModel):
    """Configuration model for dockingpp."""

    seed: int = 7
    device: str = "cpu"
    generations: int = 5
    pop_size: int = 20
    topk: int = 5
    num_atoms: int = 10
    max_trans: float = 5.0
    max_rot_deg: float = 25.0
    sw_interval: int = 5
    sw_max_iter: int = 50
    sw_patience: int = 10
    top_frac_sw: float = 0.2
    cheap_weights: Dict[str, float] = Field(default_factory=dict)
    expensive_every: int = 0
    expensive_topk: int = 0

    class Config:
        extra = "allow"


def _dummy_inputs() -> tuple[Any, Any, list[Pocket]]:
    center = np.zeros(3, dtype=float)
    pocket = Pocket(center=center, radius=5.0)
    return {"dummy": True}, {"dummy": True}, [pocket]


def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Run the docking pipeline."""

    np.random.seed(cfg.seed)
    if receptor_path == "__dummy__" and peptide_path == "__dummy__":
        receptor, peptide, pockets = _dummy_inputs()
    else:
        receptor = load_receptor(receptor_path)
        peptide = load_peptide(peptide_path)
        pockets = load_pockets(receptor)

    logger = RunLogger()
    search = ABCGAVGOSSearch(cfg)
    prior_pocket = PriorNetPocket()
    prior_pose = PriorNetPose()

    result = search.search(
        receptor=receptor,
        peptide=peptide,
        pockets=pockets,
        cfg=cfg,
        score_cheap_fn=score_pose_cheap,
        score_expensive_fn=score_pose_expensive,
        prior_pocket=prior_pocket,
        prior_pose=prior_pose,
        logger=logger,
    )

    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        payload = {
            "best_score_cheap": result.best_pose.score_cheap,
            "best_score_expensive": result.best_pose.score_expensive,
            "generation": result.best_pose.meta.get("generation"),
            "config": {
                "seed": cfg.seed,
                "generations": cfg.generations,
                "pop_size": cfg.pop_size,
                "topk": cfg.topk,
                "max_trans": cfg.max_trans,
                "max_rot_deg": cfg.max_rot_deg,
            },
        }
        handle.write(json.dumps(payload, indent=2))

    logger.flush(out_dir)
    return result
