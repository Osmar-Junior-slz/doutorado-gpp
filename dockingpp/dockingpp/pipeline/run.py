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
from dockingpp.priors.pocket import PriorNetPocket, rank_pockets
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
    top_pockets: int = 3
    full_search: bool = True
    max_pockets_used: int = 8

    class Config:
        extra = "allow"


def _dummy_inputs() -> tuple[Any, Any, list[Pocket]]:
    receptor_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    pockets = [
        Pocket(id="dummy-0", center=np.array([0.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords),
        Pocket(id="dummy-1", center=np.array([10.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords),
        Pocket(id="dummy-2", center=np.array([0.0, 10.0, 0.0]), radius=5.0, coords=receptor_coords),
    ]
    receptor = {"dummy": True, "coords": receptor_coords}
    return receptor, {"dummy": True}, pockets


def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Executa o pipeline de docking."""

    np.random.seed(cfg.seed)
    # PT-BR: criamos o diretório antes do logger para permitir escrita incremental
    # do metrics.jsonl, evitando que a UI só veja progresso no final.
    os.makedirs(out_dir, exist_ok=True)
    if receptor_path == "__dummy__" and peptide_path == "__dummy__":
        receptor, peptide, pockets = _dummy_inputs()
    else:
        receptor = load_receptor(receptor_path)
        peptide = load_peptide(peptide_path)
        pockets = load_pockets(
            receptor,
            cfg=cfg,
            pockets_path=getattr(cfg, "pockets_path", None),
        )

    # PT-BR: live_write=True garante métricas disponíveis durante a execução.
    # As métricas por geração incluem "generation" (0..N) para a UI calcular
    # progresso correto; o "step" permanece como contador global para séries.
    logger = RunLogger(out_dir=out_dir, live_write=True)
    cfg.expensive_logger = logger
    total_pockets = len(pockets)
    ranked = rank_pockets(receptor, pockets, peptide=peptide)
    if not ranked:
        global_pockets = [pocket for pocket in pockets if getattr(pocket, "id", None) == "global"]
        pockets = global_pockets or pockets
    elif not getattr(cfg, "full_search", True):
        # PT-BR: o erro anterior ocorria quando o "reduced" ainda varria todos
        # os pockets na busca. Aqui limitamos explicitamente aos top_pockets,
        # garantindo que o espaço de busca seja reduzido de fato.
        top_pockets = int(getattr(cfg, "top_pockets", len(ranked)) or 0)
        if top_pockets <= 0:
            pockets = [pocket for pocket, _ in ranked]
        elif total_pockets > top_pockets:
            pockets = [pocket for pocket, _ in ranked[:top_pockets]]
        else:
            pockets = [pocket for pocket, _ in ranked]
    else:
        max_pockets_used = int(getattr(cfg, "max_pockets_used", 8) or 0)
        if max_pockets_used <= 0:
            max_pockets_used = len(ranked)
        pockets = [pocket for pocket, _ in ranked[:max_pockets_used]]

    # PT-BR: métricas globais de seleção. "n_pockets_total" é o total detectado,
    # "n_pockets_used" é quantos realmente foram passados para a busca, e
    # "reduction_ratio" = 1 - used/total (deve ser > 0 no modo reduced).
    selected_pockets = len(pockets)
    logger.log_metric("total_pockets", float(total_pockets), step=0)
    logger.log_metric("selected_pockets", float(selected_pockets), step=0)
    logger.log_global_metrics(total_pockets, selected_pockets)
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
    mode_label = "full" if getattr(cfg, "full_search", True) else "reduced"
    logger.flush_timeseries(out_dir, mode=mode_label)
    return result
