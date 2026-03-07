"""Pipeline entrypoints."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field

from dockingpp.data.io import load_peptide, load_pockets, load_receptor
from dockingpp.data.structs import Pocket, RunResult
from dockingpp.pipeline.logging import AuditTracer, RunLogger
from dockingpp.pipeline.scan import (
    build_receptor_kdtree,
    scan_pocket_feasibility,
    select_pockets_from_scan,
)
from dockingpp.priors.pocket import PriorNetPocket, rank_pockets
from dockingpp.priors.pose import PriorNetPose
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.scoring.expensive import score_pose_expensive
from dockingpp.search.abc_ga_vgos import ABCGAVGOSSearch
from dockingpp.utils.debug_logger import DebugLogger


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
    expensive_topk: Optional[int] = None
    top_pockets: int = 3
    full_search: bool = True
    max_pockets_used: int = 8
    search_space_mode: str = "full"
    budget_policy: str = "split"
    debug_log_enabled: bool = False
    debug_log_path: Optional[str] = None
    debug_log_level: str = "INFO"
    debug_enabled: bool = True
    debug_level: str = "AUDIT"
    debug_dirname: str = "debug"
    trace_generation_interval: int = 1
    trace_score_interval: int = 1
    trace_write_jsonl: bool = True

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


def _extract_coords(receptor: Any) -> np.ndarray:
    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = getattr(receptor, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _build_global_pocket(receptor: Any, cfg: Config) -> Pocket:
    coords = _extract_coords(receptor)
    if coords.size:
        center = coords.mean(axis=0)
        deltas = coords - center.reshape(1, 3)
        max_dist = float(np.max(np.linalg.norm(deltas, axis=1)))
    else:
        center = np.zeros(3, dtype=float)
        max_dist = 0.0
    pocket_margin = float(getattr(cfg, "pocket_margin", 2.0))
    radius = max_dist + pocket_margin
    return Pocket(
        id="global",
        center=center,
        radius=radius,
        coords=coords,
        meta={"coords": coords},
    )


def _cfg_value(cfg: Optional[Any], key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_search_space_mode(search_space_mode: str | None, full_search: bool) -> str:
    raw = (search_space_mode or "").strip().lower()
    if raw in {"full", "reduced"}:
        return raw
    if raw == "global":
        return "full"
    if raw == "pockets":
        return "reduced"
    return "full" if full_search else "reduced"


def _build_run_payload(
    run_id: str,
    mode: str,
    search_space_mode: str,
    runtime_sec: float,
    total_pockets: int,
    selected_pockets: int,
    best_score_cheap: float | None,
    best_score_expensive: float | None,
    best_pose_id: str | None,
    config_resolved_subset: dict[str, Any],
    pocketing_time: float,
    scan_time: float,
    search_time: float,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "mode": mode,
        "run_id": run_id,
        "best_score_cheap": best_score_cheap,
        "best_score_expensive": best_score_expensive,
        "best_pose_id": best_pose_id,
        "n_pockets_detected": total_pockets,
        "n_pockets_used": selected_pockets,
        "search_space_mode": search_space_mode,
        "runtime_sec": runtime_sec,
        "config_resolved_subset": config_resolved_subset,
        "timing": {
            "total_s": runtime_sec,
            "scoring_cheap_s": None,
            "scoring_expensive_s": None,
            "pocketing_s": pocketing_time,
            "scan_s": scan_time,
            "search_s": search_time,
        },
    }


def _allocate_split_budget(total_generations: int, total_pop_size: int, n_pockets: int) -> list[tuple[int, int]]:
    if n_pockets <= 0:
        return []
    total_eval = max(1, int(total_generations) * int(total_pop_size))
    base = total_eval // n_pockets
    rem = total_eval % n_pockets
    budgets: list[tuple[int, int]] = []
    for idx in range(n_pockets):
        eval_budget = base + (1 if idx < rem else 0)
        pop_size = max(1, min(int(total_pop_size), int(eval_budget)))
        generations = max(1, int(eval_budget) // pop_size)
        budgets.append((generations, pop_size))
    return budgets


def _execute_single_run(
    cfg: Config,
    receptor: Any,
    peptide: Any,
    pockets: list[Pocket],
    out_dir: str,
    run_id: str,
    receptor_path: str,
    peptide_path: str,
    search_space_mode: str,
    total_pockets: int,
    selected_pockets: int,
    pocketing_time: float,
    scan_time: float,
    scan_params: dict[str, Any],
    scan_results: dict[str, dict[str, float]],
    selected_pocket_ids: list[str],
    tracer: AuditTracer | None = None,
    pocket_id: str | None = None,
) -> tuple[RunResult, dict[str, Any], RunLogger]:
    os.makedirs(out_dir, exist_ok=True)
    logger = RunLogger(out_dir=out_dir, live_write=True)
    cfg.expensive_logger = logger
    search = ABCGAVGOSSearch(cfg)
    prior_pocket = PriorNetPocket()
    prior_pose = PriorNetPose()

    if tracer is not None:
        tracer.event(stage="search", substage="start", event_type="search_started", payload={"pocket_id": pocket_id, "engine_name": "ABCGAVGOSSearch", "generations": int(cfg.generations), "pop_size": int(cfg.pop_size)}, engine="ABCGAVGOSSearch", pocket_id=pocket_id, level="BASIC")
    start_search = time.perf_counter()
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
    end_search = time.perf_counter()
    if tracer is not None:
        tracer.event(stage="search", substage="end", event_type="generation_completed", payload={"pocket_id": pocket_id, "runtime_sec": float(end_search-start_search)}, engine="ABCGAVGOSSearch", pocket_id=pocket_id, level="TRACE")

    config_resolved_subset = {
        "seed": cfg.seed,
        "generations": cfg.generations,
        "pop_size": cfg.pop_size,
        "topk": cfg.topk,
        "search_space_mode": search_space_mode,
        "full_search": bool(getattr(cfg, "full_search", True)),
        "top_pockets": int(getattr(cfg, "top_pockets", 0) or 0),
        "max_pockets_used": int(getattr(cfg, "max_pockets_used", 0) or 0),
        "expensive_every": int(getattr(cfg, "expensive_every", 0) or 0),
        "expensive_topk": getattr(cfg, "expensive_topk", None),
        "debug_log_enabled": bool(getattr(cfg, "debug_log_enabled", False)),
        "debug_log_path": getattr(cfg, "debug_log_path", None),
        "debug_log_level": getattr(cfg, "debug_log_level", None),
        "scan": scan_params,
        "budget_policy": getattr(cfg, "budget_policy", "split"),
    }
    best_pose_id = result.best_pose.meta.get("pose_id") or result.best_pose.meta.get("id")
    runtime_sec = pocketing_time + scan_time + (end_search - start_search)

    payload = _build_run_payload(
        run_id=run_id,
        mode="single",
        search_space_mode=search_space_mode,
        runtime_sec=runtime_sec,
        total_pockets=total_pockets,
        selected_pockets=selected_pockets,
        best_score_cheap=result.best_pose.score_cheap,
        best_score_expensive=result.best_pose.score_expensive,
        best_pose_id=best_pose_id,
        config_resolved_subset=config_resolved_subset,
        pocketing_time=pocketing_time,
        scan_time=scan_time,
        search_time=end_search - start_search,
    )
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
    if tracer is not None:
        tracer.artifact_written(result_path)

    logger.flush(out_dir)
    logger.flush_timeseries(out_dir, mode=search_space_mode)
    _write_summary(
        out_dir=out_dir,
        run_id=run_id,
        mode="single",
        receptor_path=receptor_path,
        peptide_path=peptide_path,
        search_space_mode=search_space_mode,
        runtime_sec=runtime_sec,
        search_time_sec=end_search - start_search,
        pocketing_sec=pocketing_time,
        scan_sec=scan_time,
        total_pockets=total_pockets,
        selected_pockets=selected_pockets,
        best_score_cheap=result.best_pose.score_cheap,
        best_score_expensive=result.best_pose.score_expensive,
        best_pose_pocket_id=result.best_pose.meta.get("pocket_id"),
        config_resolved_subset=config_resolved_subset,
        records=logger.records,
        pockets=pockets,
        scan_params=scan_params,
        scan_by_pocket=scan_results,
        selected_pocket_ids=selected_pocket_ids,
    )
    if tracer is not None:
        tracer.artifact_written(os.path.join(out_dir, "summary.json"))
        tracer.event(stage="summary", event_type="run_finished", payload={"status": "success", "best_score_cheap": result.best_pose.score_cheap, "best_score_expensive": result.best_pose.score_expensive}, level="BASIC", pocket_id=pocket_id)
    return result, payload, logger


def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Executa o pipeline de docking."""

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    np.random.seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    debug_log_enabled = bool(getattr(cfg, "debug_log_enabled", False))
    debug_log_path = getattr(cfg, "debug_log_path", None) or os.path.join(out_dir, "debug", "debug.jsonl")
    debug_log_level = str(getattr(cfg, "debug_log_level", "INFO"))
    debug_logger = DebugLogger(enabled=debug_log_enabled, path=debug_log_path, level=debug_log_level)
    debug_logger.run_id = run_id
    cfg.debug_logger = debug_logger
    tracer = AuditTracer(
        out_dir=out_dir,
        run_id=run_id,
        debug_enabled=bool(getattr(cfg, "debug_enabled", True)),
        debug_level=str(getattr(cfg, "debug_level", "AUDIT")),
        debug_dirname=str(getattr(cfg, "debug_dirname", "debug")),
        search_space_mode=str(getattr(cfg, "search_space_mode", "full")),
    )
    cfg.audit_tracer = tracer
    tracer.start_run(
        {
            "receptor_path": receptor_path,
            "peptide_path": peptide_path,
            "out_dir": out_dir,
            "seed": int(cfg.seed),
            "debug_enabled": bool(getattr(cfg, "debug_enabled", True)),
            "debug_level": str(getattr(cfg, "debug_level", "AUDIT")),
        }
    )

    try:
        if receptor_path == "__dummy__" and peptide_path == "__dummy__":
            receptor, peptide, dummy_pockets = _dummy_inputs()
        else:
            receptor = load_receptor(receptor_path)
            peptide = load_peptide(peptide_path)
            dummy_pockets = []
        tracer.event(stage="io", event_type="inputs_loaded", payload={"receptor_atoms": int(_extract_coords(receptor).shape[0]), "peptide_atoms": int(_extract_coords(peptide).shape[0])}, level="BASIC")

        search_space_mode = _normalize_search_space_mode(
            getattr(cfg, "search_space_mode", None),
            bool(getattr(cfg, "full_search", True)),
        )
        cfg.search_space_mode = search_space_mode
        cfg.full_search = search_space_mode == "full"
        tracer.search_space_mode = search_space_mode
        tracer.event(stage="config", event_type="config_loaded", substage="raw", payload={"requested_mode": getattr(cfg, "search_space_mode", None), "budget_policy": getattr(cfg, "budget_policy", "split")}, level="BASIC")
        tracer.event(stage="config", event_type="config_normalized", substage="mode", payload={"search_space_mode": search_space_mode, "full_search": bool(cfg.full_search), "compare_policy": "best_pocket_vs_full"}, level="BASIC", decision=True)
        debug_logger.log({"type": "config_resolved", "search_space_mode": search_space_mode})

        pocketing_start = time.perf_counter()
        if search_space_mode == "full":
            pockets = [_build_global_pocket(receptor, cfg)]
            total_pockets = 1
        else:
            pockets = dummy_pockets or load_pockets(
                receptor,
                cfg=cfg,
                pockets_path=getattr(cfg, "pockets_path", None),
            )
            ranked = rank_pockets(receptor, pockets, peptide=peptide, debug_logger=debug_logger)
            pockets = [p for p, _ in ranked] if ranked else pockets
            total_pockets = len(pockets)
            top_pockets = int(getattr(cfg, "top_pockets", len(pockets)) or len(pockets))
            if top_pockets > 0:
                pockets = pockets[:top_pockets]
        pocketing_time = time.perf_counter() - pocketing_start
        tracer.event(stage="pocket_filter", event_type="pocket_selected", payload={"selected_pockets": [str(p.id) for p in pockets], "n_pockets_total": int(total_pockets)}, level="BASIC", decision=True)
        debug_logger.log({"type": "pocket_selection", "selected": [str(p.id) for p in pockets], "n_total": int(total_pockets)})

        scan_cfg = _cfg_value(cfg, "scan", None)
        scan_enabled = bool(_cfg_value(scan_cfg, "enabled", False)) and search_space_mode == "reduced"
        scan_results: dict[str, dict[str, float]] = {}
        scan_params = {
            "enabled": scan_enabled,
            "max_clash_ratio": _cfg_value(scan_cfg, "max_clash_ratio", None),
            "select_top_k": _cfg_value(scan_cfg, "select_top_k", None),
        }
        scan_start = time.perf_counter()
        if scan_enabled and pockets:
            receptor_coords = _extract_coords(receptor)
            peptide_coords = _extract_coords(peptide)
            receptor_kdtree = build_receptor_kdtree(receptor_coords)
            rng = np.random.default_rng(cfg.seed + int(_cfg_value(scan_cfg, "seed_offset", 0) or 0))
            for pocket in pockets:
                scan_results[str(pocket.id)] = scan_pocket_feasibility(receptor_kdtree, peptide_coords, pocket, scan_cfg, rng)
            pockets = select_pockets_from_scan(pockets, scan_results, _cfg_value(scan_cfg, "select_top_k", None))
        scan_time = time.perf_counter() - scan_start

        if search_space_mode == "full":
            result, _, logger_single = _execute_single_run(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=pockets,
                out_dir=out_dir,
                run_id=run_id,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="full",
                total_pockets=total_pockets,
                selected_pockets=len(pockets),
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results=scan_results,
                selected_pocket_ids=[str(p.id) for p in pockets],
                tracer=tracer,
                pocket_id="global",
            )
            _write_debug_summary(tracer, {
                "success": True,
                "requested_mode": "full",
                "executed_mode": "full",
                "search_space_mode": "full",
                "budget_policy": str(getattr(cfg, "budget_policy", "split")),
                "compare_policy": "best_pocket_vs_full",
                "fallback_to_full": False,
                "fallback_reason": None,
                "total_runtime_sec": None,
                "total_n_eval": int(sum(float(r.get("value", 0.0)) for r in logger_single.records if r.get("name") == "n_eval")),
                "best_score_cheap": result.best_pose.score_cheap,
                "best_score_expensive": result.best_pose.score_expensive,
                "best_pocket_id": result.best_pose.meta.get("pocket_id"),
                "n_pockets_total": total_pockets,
                "n_pockets_used": len(pockets),
                "total_eval_budget_requested": int(cfg.generations) * int(cfg.pop_size),
                "total_eval_budget_assigned": int(cfg.generations) * int(cfg.pop_size),
                "budget_delta": 0,
                "budget_rounding_applied": False,
                "warnings_count": tracer.warnings_count,
                "errors_count": tracer.errors_count,
                "selected_pockets": [str(p.id) for p in pockets],
                "rejected_pockets": [],
            })
            return result

        max_clash_ratio = float(_cfg_value(scan_cfg, "max_clash_ratio", 0.02) or 0.02)
        feasible_pockets: list[tuple[int, Pocket]] = []
        rejected: list[dict[str, Any]] = []
        for idx, pocket in enumerate(pockets):
            metrics = scan_results.get(str(pocket.id), {})
            feasible_fraction = float(metrics.get("feasible_fraction", 1.0))
            clash_ratio_best = float(metrics.get("clash_ratio_best", 0.0))
            if feasible_fraction <= 0.0:
                rejected.append({"pocket_id": pocket.id, "reason": "feasible_fraction<=0.0"})
                tracer.event(stage="pocket_filter", event_type="pocket_rejected", payload={"reason": "feasible_fraction<=0.0", "feasible_fraction": feasible_fraction}, pocket_id=str(pocket.id), level="TRACE", decision=True)
                continue
            if clash_ratio_best > max_clash_ratio:
                rejected.append({"pocket_id": pocket.id, "reason": "clash_ratio_best>max_clash_ratio"})
                tracer.event(stage="pocket_filter", event_type="pocket_rejected", payload={"reason": "clash_ratio_best>max_clash_ratio", "clash_ratio_best": clash_ratio_best, "max_clash_ratio": max_clash_ratio}, pocket_id=str(pocket.id), level="TRACE", decision=True)
                continue
            feasible_pockets.append((idx, pocket))

        if not feasible_pockets:
            tracer.event(stage="budget", event_type="budget_split", payload={"total_eval_budget_requested": int(cfg.generations) * int(cfg.pop_size), "n_pockets": 0, "allocations": []}, level="TRACE", decision=True)
            tracer.event(stage="pocket_filter", event_type="fallback_triggered", payload={"reason": "no_feasible_pocket"}, level="BASIC", decision=True)
            fallback_dir = os.path.join(out_dir, "fallback_full")
            full_cfg = Config(**cfg.model_dump())
            full_cfg.search_space_mode = "full"
            full_cfg.full_search = True
            full_pockets = [_build_global_pocket(receptor, full_cfg)]
            result, full_payload, full_logger = _execute_single_run(
                cfg=full_cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=full_pockets,
                out_dir=fallback_dir,
                run_id=f"{run_id}-fallback",
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="full",
                total_pockets=1,
                selected_pockets=1,
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results=scan_results,
                selected_pocket_ids=["global"],
                tracer=tracer,
                pocket_id="global",
            )
            requested_budget = int(cfg.generations) * int(cfg.pop_size)
            assigned_budget = int(full_cfg.generations) * int(full_cfg.pop_size)
            parent_summary = {
                "schema_version": "2.0",
                "mode": "reduced_aggregate",
                "search_space_mode": "reduced",
                "budget_policy": str(getattr(cfg, "budget_policy", "split")),
                "compare_policy": "best_pocket_vs_full",
                "fallback_to_full": True,
                "fallback_from": "reduced",
                "fallback_reason": "no_feasible_pocket",
                "executed_mode": "full",
                "rejected_pockets": rejected,
                "total_runtime_sec": float(full_payload.get("runtime_sec", 0.0)),
                "total_n_eval": int(sum(float(r.get("value", 0.0)) for r in full_logger.records if r.get("name") == "n_eval")),
                "total_eval_budget_requested": requested_budget,
                "total_eval_budget_assigned": assigned_budget,
                "budget_delta": int(assigned_budget - requested_budget),
                "budget_rounding_applied": bool(assigned_budget != requested_budget),
                "n_pockets_total": total_pockets,
                "n_pockets_used": 0,
                "selected_pockets": [],
                "best_pocket_id": "global",
                "best_over_pockets_cheap": result.best_pose.score_cheap,
                "best_over_pockets_expensive": result.best_pose.score_expensive,
                "per_pocket_results": [],
                "fallback_full_outdir": fallback_dir,
            }
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(parent_summary, indent=2))
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(parent_summary, indent=2))
            _write_debug_summary(tracer, {
                "success": True,
                "requested_mode": "reduced",
                "executed_mode": "full",
                "search_space_mode": "reduced",
                "budget_policy": str(getattr(cfg, "budget_policy", "split")),
                "compare_policy": "best_pocket_vs_full",
                "fallback_to_full": True,
                "fallback_reason": "no_feasible_pocket",
                "total_runtime_sec": float(full_payload.get("runtime_sec", 0.0)),
                "total_n_eval": int(sum(float(r.get("value", 0.0)) for r in full_logger.records if r.get("name") == "n_eval")),
                "best_score_cheap": result.best_pose.score_cheap,
                "best_score_expensive": result.best_pose.score_expensive,
                "best_pocket_id": "global",
                "n_pockets_total": total_pockets,
                "n_pockets_used": 0,
                "total_eval_budget_requested": requested_budget,
                "total_eval_budget_assigned": assigned_budget,
                "budget_delta": int(assigned_budget - requested_budget),
                "budget_rounding_applied": bool(assigned_budget != requested_budget),
                "warnings_count": tracer.warnings_count,
                "errors_count": tracer.errors_count,
                "selected_pockets": [],
                "rejected_pockets": rejected,
            })
            return result

        budget_policy = str(getattr(cfg, "budget_policy", "split") or "split").lower()
        if budget_policy not in {"split", "replicated"}:
            budget_policy = "split"
        budgets = _allocate_split_budget(cfg.generations, cfg.pop_size, len(feasible_pockets))
        tracer.event(stage="budget", event_type="budget_split", payload={"total_eval_budget_requested": int(cfg.generations) * int(cfg.pop_size), "n_pockets": len(feasible_pockets), "allocations": [{"pocket_id": str(p.id), "generations": int(b[0]), "pop_size": int(b[1])} for (_, p), b in zip(feasible_pockets, budgets)]}, level="TRACE", decision=True)

        per_pocket_results = []
        total_runtime_sec = 0.0
        total_n_eval = 0
        total_eval_budget_requested = int(cfg.generations) * int(cfg.pop_size)
        total_eval_budget_assigned = 0
        budget_rounding_applied = False
        best_result: RunResult | None = None
        best_cheap = float("-inf")
        best_expensive = None
        best_pocket_id = None

        for local_idx, (original_idx, pocket) in enumerate(feasible_pockets):
            pocket_cfg = Config(**cfg.model_dump())
            pocket_cfg.search_space_mode = "reduced"
            pocket_cfg.full_search = False
            if budget_policy == "split":
                pocket_cfg.generations, pocket_cfg.pop_size = budgets[local_idx]
            assigned_budget = int(pocket_cfg.generations) * int(pocket_cfg.pop_size)
            total_eval_budget_assigned += assigned_budget
            pocket_out_dir = os.path.join(out_dir, str(pocket.id))
            result, payload, logger = _execute_single_run(
                cfg=pocket_cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=[pocket],
                out_dir=pocket_out_dir,
                run_id=f"{run_id}-pocket-{pocket.id}",
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="reduced",
                total_pockets=total_pockets,
                selected_pockets=1,
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results={str(pocket.id): scan_results.get(str(pocket.id), {})},
                selected_pocket_ids=[str(pocket.id)],
                tracer=tracer,
                pocket_id=str(pocket.id),
            )
            runtime = float(payload.get("runtime_sec", 0.0))
            n_eval = int(sum(float(r.get("value", 0.0)) for r in logger.records if r.get("name") == "n_eval"))
            total_runtime_sec += runtime
            total_n_eval += n_eval
            score_cheap = result.best_pose.score_cheap
            score_expensive = result.best_pose.score_expensive
            if score_cheap is not None and score_cheap > best_cheap:
                best_cheap = score_cheap
                best_expensive = score_expensive
                best_result = result
                best_pocket_id = str(pocket.id)
            metrics = scan_results.get(str(pocket.id), {})
            per_pocket_results.append(
                {
                    "pocket_id": str(pocket.id),
                    "pocket_index": int(original_idx),
                    "runtime_sec": runtime,
                    "n_eval_total": n_eval,
                    "generations": int(pocket_cfg.generations),
                    "pop_size": int(pocket_cfg.pop_size),
                    "best_score_cheap": score_cheap,
                    "best_score_expensive": score_expensive,
                    "best_pose_id": result.best_pose.meta.get("pose_id") or result.best_pose.meta.get("id"),
                    "feasible_fraction": float(metrics.get("feasible_fraction", 0.0)),
                    "clash_ratio_best": float(metrics.get("clash_ratio_best", 0.0)),
                    "outdir": pocket_out_dir,
                    "eval_budget_assigned": assigned_budget,
                }
            )

        budget_delta = int(total_eval_budget_assigned - total_eval_budget_requested)
        budget_rounding_applied = budget_delta != 0
        parent_summary = {
            "schema_version": "2.0",
            "mode": "reduced_aggregate",
            "search_space_mode": "reduced",
            "budget_policy": budget_policy,
            "compare_policy": "best_pocket_vs_full",
            "fallback_to_full": False,
            "total_runtime_sec": total_runtime_sec,
            "total_n_eval": total_n_eval,
            "total_eval_budget_requested": total_eval_budget_requested,
            "total_eval_budget_assigned": total_eval_budget_assigned,
            "budget_delta": budget_delta,
            "budget_rounding_applied": budget_rounding_applied,
            "n_pockets_total": total_pockets,
            "n_pockets_used": len(feasible_pockets),
            "selected_pockets": [str(p.id) for _, p in feasible_pockets],
            "best_pocket_id": best_pocket_id,
            "best_over_pockets_cheap": None if best_result is None else best_result.best_pose.score_cheap,
            "best_over_pockets_expensive": best_expensive,
            "per_pocket_results": per_pocket_results,
            "rejected_pockets": rejected,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(parent_summary, indent=2))
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(parent_summary, indent=2))
        with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            for item in per_pocket_results:
                handle.write(json.dumps({"name": "pocket.best_score_cheap", "pocket_id": item["pocket_id"], "value": item["best_score_cheap"]}) + "\n")
                handle.write(json.dumps({"name": "pocket.n_eval_total", "pocket_id": item["pocket_id"], "value": item["n_eval_total"]}) + "\n")
        with open(os.path.join(out_dir, "metrics.timeseries.jsonl"), "w", encoding="utf-8") as handle:
            for idx, item in enumerate(per_pocket_results):
                handle.write(json.dumps({"step": idx, "pocket_id": item["pocket_id"], "best_score_cheap": item["best_score_cheap"], "n_eval_cumulative": sum(x["n_eval_total"] for x in per_pocket_results[: idx + 1])}) + "\n")

        _write_debug_summary(tracer, {
            "success": True,
            "requested_mode": "reduced",
            "executed_mode": "reduced",
            "search_space_mode": "reduced",
            "budget_policy": budget_policy,
            "compare_policy": "best_pocket_vs_full",
            "fallback_to_full": False,
            "fallback_reason": None,
            "total_runtime_sec": total_runtime_sec,
            "total_n_eval": total_n_eval,
            "best_score_cheap": None if best_result is None else best_result.best_pose.score_cheap,
            "best_score_expensive": best_expensive,
            "best_pocket_id": best_pocket_id,
            "n_pockets_total": total_pockets,
            "n_pockets_used": len(feasible_pockets),
            "total_eval_budget_requested": total_eval_budget_requested,
            "total_eval_budget_assigned": total_eval_budget_assigned,
            "budget_delta": budget_delta,
            "budget_rounding_applied": budget_rounding_applied,
            "warnings_count": tracer.warnings_count,
            "errors_count": tracer.errors_count,
            "selected_pockets": [str(p.id) for _, p in feasible_pockets],
            "rejected_pockets": rejected,
        })
        for item in per_pocket_results:
            pocket_trace = {
                "success": True,
                "requested_mode": "reduced",
                "executed_mode": "reduced",
                "search_space_mode": "reduced",
                "best_score_cheap": item["best_score_cheap"],
                "best_score_expensive": item["best_score_expensive"],
                "best_pocket_id": item["pocket_id"],
            }
            tracer.write_summary(pocket_trace, rel_path=f"pockets/{item['pocket_id']}/debug_summary.json")
            for rel in (f"pockets/{item['pocket_id']}/trace.jsonl", f"pockets/{item['pocket_id']}/decision_trace.jsonl"):
                abs_path = os.path.join(tracer.debug_dir, rel)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                if not os.path.exists(abs_path):
                    with open(abs_path, "w", encoding="utf-8") as handle:
                        handle.write("")
                tracer._mark_file(rel)

        if best_result is None:
            raise ValueError("Reduced aggregate produced no result.")
        return best_result
    except Exception as exc:
        tracer.error(stage="summary", message="pipeline_exception", payload={"error": type(exc).__name__})
        raise
    finally:
        manifest = {
            "requested_mode": str(getattr(cfg, "search_space_mode", "full")),
            "executed_mode": str(getattr(cfg, "search_space_mode", "full")),
            "search_space_mode": str(getattr(cfg, "search_space_mode", "full")),
            "budget_policy": str(getattr(cfg, "budget_policy", "split")),
            "compare_policy": "best_pocket_vs_full",
            "fallback_to_full": False,
            "fallback_reason": None,
            "debug_enabled": bool(getattr(cfg, "debug_enabled", True)),
            "debug_level": str(getattr(cfg, "debug_level", "AUDIT")),
            "out_dir": out_dir,
            "debug_dir": tracer.debug_dir,
        }
        tracer.finish_run(manifest=manifest, status_final="finished")
        debug_logger.close()



def _write_debug_summary(tracer: AuditTracer, payload: dict[str, Any], rel_path: str = "debug_summary.json") -> None:
    if not tracer.enabled:
        return
    data = {
        "schema_version": "1.0",
        "run_id": tracer.run_id,
        **payload,
    }
    tracer.write_summary(data, rel_path=rel_path)

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_field(record: Dict[str, Any], key: str) -> Any:
    if key in record:
        return record.get(key)
    extras = record.get("extras")
    if isinstance(extras, dict):
        return extras.get(key)
    return None


def _write_summary(
    out_dir: str,
    run_id: str,
    mode: str,
    receptor_path: str,
    peptide_path: str,
    search_space_mode: str,
    runtime_sec: float,
    search_time_sec: float,
    pocketing_sec: float,
    scan_sec: float,
    total_pockets: int,
    selected_pockets: int,
    best_score_cheap: float | None,
    best_score_expensive: float | None,
    best_pose_pocket_id: str | None,
    config_resolved_subset: dict[str, Any],
    records: list[dict[str, Any]],
    pockets: list[Pocket],
    scan_params: dict[str, Any],
    scan_by_pocket: dict[str, dict[str, float]],
    selected_pocket_ids: list[str],
) -> None:
    expensive_ran = 0.0
    expensive_skipped = 0.0
    n_eval_total = 0.0
    best_by_pocket: dict[int, float] = {}

    for record in records:
        name = record.get("name")
        value = _safe_float(record.get("value"))
        if name == "expensive_ran" and value is not None:
            expensive_ran += value
        elif name == "expensive_skipped" and value is not None:
            expensive_skipped += value
        elif name == "n_eval" and value is not None:
            n_eval_total += value
        elif name == "best_score" and value is not None:
            pocket_index = _record_field(record, "pocket_index")
            if pocket_index is None:
                continue
            pocket_idx = int(pocket_index)
            current = best_by_pocket.get(pocket_idx)
            if current is None or value > current:
                best_by_pocket[pocket_idx] = value

    best_cheap_by_pocket = []
    for pocket_idx in sorted(best_by_pocket):
        pocket_id = pockets[pocket_idx].id if pocket_idx < len(pockets) else str(pocket_idx)
        best_cheap_by_pocket.append(
            {"pocket_id": pocket_id, "best_score_cheap": best_by_pocket[pocket_idx]}
        )

    best_expensive_by_pocket = []
    if best_score_expensive is not None and best_pose_pocket_id is not None:
        best_expensive_by_pocket.append(
            {"pocket_id": best_pose_pocket_id, "best_score_expensive": best_score_expensive}
        )

    reduction_ratio = 0.0
    if total_pockets > 0:
        reduction_ratio = max(0.0, 1.0 - (float(selected_pockets) / float(total_pockets)))

    input_id = getattr(config_resolved_subset, "input_id", None)
    if input_id is None and isinstance(config_resolved_subset, dict):
        input_id = config_resolved_subset.get("input_id")
    if input_id is None:
        receptor_name = os.path.basename(receptor_path) if receptor_path else ""
        peptide_name = os.path.basename(peptide_path) if peptide_path else ""
        if receptor_name and peptide_name:
            input_id = f"{receptor_name}__{peptide_name}"
        elif receptor_name:
            input_id = receptor_name
        elif peptide_name:
            input_id = peptide_name
        else:
            input_id = None
    complex_id = None
    if isinstance(config_resolved_subset, dict):
        complex_id = config_resolved_subset.get("complex_id")
        if complex_id is None:
            complex_id = config_resolved_subset.get("input_id")

    expensive_every = int(getattr(config_resolved_subset, "expensive_every", 0) or 0)
    if isinstance(config_resolved_subset, dict):
        expensive_every = int(config_resolved_subset.get("expensive_every", 0) or 0)
    expensive_topk = None
    if isinstance(config_resolved_subset, dict):
        expensive_topk = config_resolved_subset.get("expensive_topk")
    expensive_enabled = expensive_every > 0
    expensive_policy = {
        "every": expensive_every,
        "topk": expensive_topk,
    }

    summary_payload = {
        "schema_version": "2.0",
        "run_id": run_id,
        "complex_id": complex_id,
        "input_id": input_id,
        "seed": config_resolved_subset.get("seed") if isinstance(config_resolved_subset, dict) else None,
        "search_space_mode": search_space_mode,
        "runtime_sec": runtime_sec,
        "search_time_sec": search_time_sec,
        "pocketing_sec": pocketing_sec,
        "scan_sec": scan_sec,
        "n_eval_total": int(n_eval_total),
        "n_pockets_total": total_pockets,
        "n_pockets_used": selected_pockets,
        "reduction_ratio": reduction_ratio,
        "best_score_cheap": best_score_cheap,
        "best_score_expensive": best_score_expensive,
        "expensive_enabled": expensive_enabled,
        "expensive_policy": expensive_policy,
        "mode": mode,
        "n_pockets_detected": total_pockets,
        "expensive_ran_count": int(expensive_ran),
        "expensive_skipped_count": int(expensive_skipped),
        "best_cheap_by_pocket": best_cheap_by_pocket,
        "best_expensive_by_pocket": best_expensive_by_pocket,
        "config_resolved_subset": config_resolved_subset,
        "scan": scan_params,
        "scan_by_pocket": scan_by_pocket,
        "selected_pockets": selected_pocket_ids,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary_payload, indent=2))
