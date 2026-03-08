"""Execução da busca do pipeline após seleção de bolsões."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ContextoExecucaoBusca:
    """Dados necessários para executar a busca após a seleção."""

    search_space_mode: str
    modo_legado_pockets: bool
    pockets: list[Any]
    total_pockets: int
    scan_results: dict[str, dict[str, Any]]
    scan_params: dict[str, Any]
    pocketing_time: float
    scan_time: float
    feasible_pockets: list[tuple[int, Any]]
    rejected: list[dict[str, Any]]


class ExecutorBuscaPipeline:
    """Executa caminhos full/reduced preservando comportamento existente."""

    def __init__(
        self,
        *,
        executar_execucao_unica: Callable[..., Any],
        escrever_debug_summary: Callable[..., None],
        alocar_orcamento: Callable[[int, int, int], list[tuple[int, int]]],
        construir_bolsao_global: Callable[..., Any],
        classe_config: Any,
    ) -> None:
        self._executar_execucao_unica = executar_execucao_unica
        self._escrever_debug_summary = escrever_debug_summary
        self._alocar_orcamento = alocar_orcamento
        self._construir_bolsao_global = construir_bolsao_global
        self._classe_config = classe_config

    def executar(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        receptor_path: str,
        peptide_path: str,
        out_dir: str,
        run_id: str,
        tracer: Any,
        contexto: ContextoExecucaoBusca,
    ) -> Any:
        """Executa a busca no modo correspondente ao contexto recebido."""

        search_space_mode = contexto.search_space_mode
        modo_legado_pockets = contexto.modo_legado_pockets
        pockets = contexto.pockets
        total_pockets = contexto.total_pockets
        scan_results = contexto.scan_results
        scan_params = contexto.scan_params
        pocketing_time = contexto.pocketing_time
        scan_time = contexto.scan_time
        feasible_pockets = contexto.feasible_pockets
        rejected = contexto.rejected

        if search_space_mode == "full":
            result, _, logger_single = self._executar_execucao_unica(
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
            self._escrever_debug_summary(
                tracer,
                {
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
                },
            )
            return result

        if search_space_mode == "reduced" and modo_legado_pockets:
            result, _, logger_single = self._executar_execucao_unica(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=pockets,
                out_dir=out_dir,
                run_id=run_id,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="reduced",
                total_pockets=total_pockets,
                selected_pockets=len(pockets),
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results=scan_results,
                selected_pocket_ids=[str(p.id) for p in pockets],
                tracer=tracer,
                pocket_id="legacy_pockets",
            )
            self._escrever_debug_summary(
                tracer,
                {
                    "success": True,
                    "requested_mode": "pockets",
                    "executed_mode": "reduced",
                    "search_space_mode": "reduced",
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
                },
            )
            return result

        if not feasible_pockets:
            tracer.event(stage="budget", event_type="budget_split", payload={"total_eval_budget_requested": int(cfg.generations) * int(cfg.pop_size), "n_pockets": 0, "allocations": []}, level="TRACE", decision=True)
            tracer.event(stage="pocket_filter", event_type="fallback_triggered", payload={"reason": "no_feasible_pocket"}, level="BASIC", decision=True)
            fallback_dir = os.path.join(out_dir, "fallback_full")
            full_cfg = self._classe_config(**cfg.model_dump())
            full_cfg.search_space_mode = "full"
            full_cfg.full_search = True
            full_pockets = [self._construir_bolsao_global(receptor, full_cfg)]
            result, full_payload, full_logger = self._executar_execucao_unica(
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
                "run_id": run_id,
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
            self._escrever_debug_summary(
                tracer,
                {
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
                },
            )
            return result

        budget_policy = str(getattr(cfg, "budget_policy", "split") or "split").lower()
        if budget_policy not in {"split", "replicated"}:
            budget_policy = "split"
        budgets = self._alocar_orcamento(cfg.generations, cfg.pop_size, len(feasible_pockets))
        tracer.event(stage="budget", event_type="budget_split", payload={"total_eval_budget_requested": int(cfg.generations) * int(cfg.pop_size), "n_pockets": len(feasible_pockets), "allocations": [{"pocket_id": str(p.id), "generations": int(b[0]), "pop_size": int(b[1])} for (_, p), b in zip(feasible_pockets, budgets)]}, level="TRACE", decision=True)

        per_pocket_results = []
        total_runtime_sec = 0.0
        total_n_eval = 0
        total_eval_budget_requested = int(cfg.generations) * int(cfg.pop_size)
        total_eval_budget_assigned = 0
        budget_rounding_applied = False
        best_result = None
        best_cheap = float("-inf")
        best_expensive = None
        best_pocket_id = None

        for local_idx, (original_idx, pocket) in enumerate(feasible_pockets):
            pocket_cfg = self._classe_config(**cfg.model_dump())
            pocket_cfg.search_space_mode = "reduced"
            pocket_cfg.full_search = False
            if budget_policy == "split":
                pocket_cfg.generations, pocket_cfg.pop_size = budgets[local_idx]
            assigned_budget = int(pocket_cfg.generations) * int(pocket_cfg.pop_size)
            total_eval_budget_assigned += assigned_budget
            pocket_out_dir = os.path.join(out_dir, str(pocket.id))
            result, payload, logger = self._executar_execucao_unica(
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
                best_result = result
                best_pocket_id = str(pocket.id)
            if score_expensive is not None and (best_expensive is None or score_expensive > best_expensive):
                best_expensive = score_expensive
            per_pocket_results.append(
                {
                    "pocket_index": int(original_idx),
                    "pocket_id": str(pocket.id),
                    "best_score_cheap": score_cheap,
                    "best_score_expensive": score_expensive,
                    "runtime_sec": runtime,
                    "n_eval_total": n_eval,
                    "alloc_generations": int(pocket_cfg.generations),
                    "alloc_pop_size": int(pocket_cfg.pop_size),
                    "alloc_eval_budget": assigned_budget,
                }
            )

        budget_delta = int(total_eval_budget_assigned - total_eval_budget_requested)
        budget_rounding_applied = budget_delta != 0
        parent_summary = {
            "schema_version": "2.0",
            "mode": "reduced_aggregate",
            "run_id": run_id,
            "search_space_mode": "reduced",
            "budget_policy": budget_policy,
            "compare_policy": "best_pocket_vs_full",
            "fallback_to_full": False,
            "fallback_reason": None,
            "executed_mode": "reduced",
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

        self._escrever_debug_summary(
            tracer,
            {
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
            },
        )
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
