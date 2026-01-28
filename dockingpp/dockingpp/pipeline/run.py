"""Entradas do pipeline principal."""

# PT-BR: este módulo coordena o pipeline PhD (scan -> detecção -> ranking -> docking).

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, Field

from dockingpp.core.deteccao_bolsoes import construir_bolso_global, detectar_bolsoes
from dockingpp.core.escaneamento_receptor import escanear_receptor
from dockingpp.core.ranqueamento_bolsoes import ranquear_bolsoes, selecionar_top_bolsoes
from dockingpp.data.io import load_peptide, load_receptor
from dockingpp.data.structs import Pocket, RunResult
from dockingpp.pipeline.logging import RunLogger
from dockingpp.priors.pocket import PriorNetPocket
from dockingpp.priors.pose import PriorNetPose
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.scoring.expensive import score_pose_expensive
from dockingpp.search.abc_ga_vgos import ABCGAVGOSSearch


class Config(BaseModel):
    """Modelo de configuração do dockingpp."""

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

    class Config:
        extra = "allow"


def _dummy_inputs() -> tuple[Any, Any]:
    """Cria inputs dummy para testes rápidos (PT-BR)."""
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
    receptor = {"dummy": True, "coords": receptor_coords}
    return receptor, {"dummy": True}


def _build_fallback_pocket(receptor: Any, cfg: Config) -> Pocket:
    """Constrói um bolso global como fallback explícito (PT-BR)."""

    if isinstance(receptor, dict):
        coords = np.asarray(receptor.get("coords", np.zeros((0, 3), dtype=float)), dtype=float)
    elif isinstance(receptor, np.ndarray):
        coords = np.asarray(receptor, dtype=float)
    else:
        coords = np.asarray(getattr(receptor, "coords", np.zeros((0, 3), dtype=float)), dtype=float)
    return construir_bolso_global(coords, cfg=cfg)


def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Executa o pipeline de docking."""

    np.random.seed(cfg.seed)
    # PT-BR: criamos o diretório antes do logger para permitir escrita incremental
    # do metrics.jsonl, evitando que a UI só veja progresso no final.
    os.makedirs(out_dir, exist_ok=True)
    if receptor_path == "__dummy__" and peptide_path == "__dummy__":
        receptor, peptide = _dummy_inputs()
    else:
        receptor = load_receptor(receptor_path)
        peptide = load_peptide(peptide_path)

    # PT-BR: live_write=True garante métricas disponíveis durante a execução.
    # As métricas por geração incluem "generation" (0..N) para a UI calcular
    # progresso correto; o "step" permanece como contador global para séries.
    logger = RunLogger(out_dir=out_dir, live_write=True)
    scan_start = time.perf_counter()
    scan = escanear_receptor(receptor, cfg=cfg)
    scan_time = time.perf_counter() - scan_start

    detection_start = time.perf_counter()
    detected = detectar_bolsoes(scan, cfg=cfg)
    detection_time = time.perf_counter() - detection_start

    pocket_fallback_used = 0
    detected_count = len(detected)
    if detected_count == 0:
        # PT-BR: fallback global explícito (somente quando detecção falha/zero).
        detected = [_build_fallback_pocket(receptor, cfg)]
        pocket_fallback_used = 1

    ranking_start = time.perf_counter()
    ranked = ranquear_bolsoes(receptor, detected, peptide=peptide)
    ranking_time = time.perf_counter() - ranking_start

    full_search = bool(getattr(cfg, "full_search", True))
    top_pockets = int(getattr(cfg, "top_pockets", len(ranked)) or 0)
    pockets = selecionar_top_bolsoes(ranked, top_pockets, full_search=full_search)

    # PT-BR: métricas globais de seleção. "n_pockets_total" é o total detectado,
    # "n_pockets_used" é quantos realmente foram passados para a busca, e
    # "reduction_ratio" = 1 - used/total (deve ser > 0 no modo reduced).
    total_pockets = detected_count
    selected_pockets = len(pockets)
    logger.log_metric("total_pockets", float(total_pockets), step=0)
    logger.log_metric("selected_pockets", float(selected_pockets), step=0)
    logger.log_metric("n_pockets_detected", float(total_pockets), step=0)
    logger.log_metric("n_pockets_selected", float(selected_pockets), step=0)
    logger.log_metric("pocket_fallback_used", float(pocket_fallback_used), step=0)
    logger.log_metric("scan_time_seconds", float(scan_time), step=0)
    logger.log_metric("pocket_detection_time_seconds", float(detection_time), step=0)
    logger.log_metric("pocket_ranking_time_seconds", float(ranking_time), step=0)
    logger.log_global_metrics(total_pockets, selected_pockets)
    search = ABCGAVGOSSearch(cfg)
    prior_pocket = PriorNetPocket()
    prior_pose = PriorNetPose()

    docking_start = time.perf_counter()
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
    docking_time = time.perf_counter() - docking_start
    logger.log_metric("docking_time_seconds", float(docking_time), step=0)

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
