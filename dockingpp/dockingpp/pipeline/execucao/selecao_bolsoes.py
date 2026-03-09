"""Seleção de bolsões para a execução do pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dockingpp.data.io import load_pockets
from dockingpp.data.structs import Pocket
from dockingpp.pipeline.scan import (
    build_receptor_kdtree,
    scan_pocket_feasibility,
    scan_pocket_feasibility_geom_kdtree,
    select_pockets_from_scan,
)
from dockingpp.priors.pocket import rank_pockets


@dataclass(frozen=True)
class ContextoSelecaoBolsoes:
    """Estado consolidado da seleção de bolsões para o pipeline."""

    search_space_mode: str
    modo_legado_pockets: bool
    candidate_pockets: list[Pocket]
    selected_pockets: list[Pocket]
    # Compatibilidade: `pockets` representa os bolsões selecionados antes do
    # filtro final de factibilidade (não os aceitos finais).
    pockets: list[Pocket]
    total_pockets: int
    scan_results: dict[str, dict[str, Any]]
    scan_params: dict[str, Any]
    pocketing_time: float
    scan_time: float
    accepted_pockets: list[tuple[int, Pocket]]
    # Compatibilidade: `feasible_pockets` permanece como alias dos aceitos.
    feasible_pockets: list[tuple[int, Pocket]]
    rejected: list[dict[str, Any]]


class SelecionadorBolsoesPipeline:
    """Encapsula seleção de bolsões preservando comportamento existente."""

    def __init__(
        self,
        *,
        normalizar_modo_busca: Callable[[str | None, bool], str],
        obter_valor_cfg: Callable[[Any, str, Any], Any],
        extrair_coords: Callable[[Any], np.ndarray],
        aplicar_reducao_condicionada: Callable[..., list[Pocket]],
    ) -> None:
        self._normalizar_modo_busca = normalizar_modo_busca
        self._obter_valor_cfg = obter_valor_cfg
        self._extrair_coords = extrair_coords
        self._aplicar_reducao_condicionada = aplicar_reducao_condicionada

    def selecionar(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        dummy_pockets: list[Pocket],
        tracer: Any,
        debug_logger: Any,
    ) -> ContextoSelecaoBolsoes:
        """Seleciona bolsões para execução mantendo semântica atual."""

        modo_requisitado_bruto = getattr(cfg, "search_space_mode", None)
        modo_legado_pockets = str(modo_requisitado_bruto or "").strip().lower() == "pockets"
        search_space_mode = self._normalizar_modo_busca(
            modo_requisitado_bruto,
            bool(getattr(cfg, "full_search", True)),
        )
        cfg.search_space_mode = search_space_mode
        cfg.full_search = search_space_mode == "full"
        tracer.search_space_mode = search_space_mode
        tracer.event(
            stage="config",
            event_type="config_loaded",
            substage="raw",
            payload={
                "requested_mode": modo_requisitado_bruto,
                "budget_policy": getattr(cfg, "budget_policy", "split"),
            },
            level="BASIC",
        )
        tracer.event(
            stage="config",
            event_type="config_normalized",
            substage="mode",
            payload={
                "search_space_mode": search_space_mode,
                "full_search": bool(cfg.full_search),
                "compare_policy": "best_pocket_vs_full",
            },
            level="BASIC",
            decision=True,
        )
        debug_logger.log({"type": "config_resolved", "search_space_mode": search_space_mode})

        pocketing_start = time.perf_counter()
        if search_space_mode == "full" and not modo_legado_pockets:
            from dockingpp.pipeline import run as run_mod

            loaded_pockets = [run_mod._build_global_pocket(receptor, cfg)]
        else:
            loaded_pockets = dummy_pockets or load_pockets(
                receptor,
                cfg=cfg,
                pockets_path=getattr(cfg, "pockets_path", None),
            )
        tracer.event(
            stage="pocket_filter",
            event_type="pockets_loaded",
            payload={"loaded_pockets": [str(p.id) for p in loaded_pockets], "n_loaded": int(len(loaded_pockets))},
            level="TRACE",
        )
        debug_logger.log(
            {
                "type": "pockets_loaded",
                "loaded": [str(p.id) for p in loaded_pockets],
                "n_loaded": int(len(loaded_pockets)),
            }
        )

        if search_space_mode == "full" and not modo_legado_pockets:
            ranked_pockets = list(loaded_pockets)
        else:
            ranked = rank_pockets(receptor, loaded_pockets, peptide=peptide, debug_logger=debug_logger)
            ranked_pockets = [p for p, _ in ranked] if ranked else list(loaded_pockets)

        total_pockets = len(ranked_pockets)
        candidate_pockets = list(ranked_pockets)
        if search_space_mode != "full":
            top_pockets = int(getattr(cfg, "top_pockets", len(candidate_pockets)) or len(candidate_pockets))
            if top_pockets > 0:
                candidate_pockets = candidate_pockets[:top_pockets]

        pocketing_time = time.perf_counter() - pocketing_start
        tracer.event(
            stage="pocket_filter",
            event_type="pockets_ranked",
            payload={"ranked_pockets": [str(p.id) for p in ranked_pockets], "n_ranked": int(len(ranked_pockets))},
            level="TRACE",
            decision=True,
        )
        debug_logger.log(
            {
                "type": "pocket_ranking",
                "ranked": [str(p.id) for p in ranked_pockets],
                "n_ranked": int(len(ranked_pockets)),
            }
        )
        tracer.event(
            stage="pocket_filter",
            event_type="pocket_selected",
            payload={"selected_pockets": [str(p.id) for p in candidate_pockets], "n_pockets_total": int(total_pockets)},
            level="BASIC",
            decision=True,
        )
        debug_logger.log(
            {
                "type": "pocket_selection",
                "selected": [str(p.id) for p in candidate_pockets],
                "n_total": int(total_pockets),
            }
        )

        scan_cfg = self._obter_valor_cfg(cfg, "scan", None)
        scan_enabled = bool(self._obter_valor_cfg(scan_cfg, "enabled", False)) and search_space_mode == "reduced"
        selector_mode = str(self._obter_valor_cfg(scan_cfg, "selector_mode", "legacy") or "legacy").strip().lower()
        if selector_mode not in {"legacy", "geom_kdtree"}:
            selector_mode = "legacy"

        scan_selected_pockets = list(candidate_pockets)
        scan_results: dict[str, dict[str, Any]] = {}
        scan_params = {
            "enabled": scan_enabled,
            "selector_mode": selector_mode,
            "max_clash_ratio": self._obter_valor_cfg(scan_cfg, "max_clash_ratio", None),
            "select_top_k": self._obter_valor_cfg(scan_cfg, "select_top_k", None),
            "reject_if_feasible_fraction_leq": self._obter_valor_cfg(scan_cfg, "reject_if_feasible_fraction_leq", 0.0),
            "reject_if_severe_clash_fraction_geq": self._obter_valor_cfg(scan_cfg, "reject_if_severe_clash_fraction_geq", 0.95),
        }
        scan_start = time.perf_counter()
        if scan_enabled and scan_selected_pockets:
            receptor_coords = self._extrair_coords(receptor)
            peptide_coords = self._extrair_coords(peptide)
            receptor_kdtree = build_receptor_kdtree(receptor_coords)
            rng = np.random.default_rng(cfg.seed + int(self._obter_valor_cfg(scan_cfg, "seed_offset", 0) or 0))
            for pocket in scan_selected_pockets:
                if selector_mode == "geom_kdtree":
                    scan_results[str(pocket.id)] = scan_pocket_feasibility_geom_kdtree(
                        receptor_kdtree,
                        receptor_coords,
                        peptide_coords,
                        pocket,
                        scan_cfg,
                        rng,
                    )
                else:
                    scan_results[str(pocket.id)] = scan_pocket_feasibility(
                        receptor_kdtree,
                        peptide_coords,
                        pocket,
                        scan_cfg,
                        rng,
                    )
            scan_selected_pockets = select_pockets_from_scan(
                scan_selected_pockets,
                scan_results,
                self._obter_valor_cfg(scan_cfg, "select_top_k", None),
                selector_mode=selector_mode,
            )
            ranking_payload = [
                {
                    "pocket_id": str(p.id),
                    "rank": int(idx),
                    "score": float(
                        scan_results.get(str(p.id), {}).get(
                            "pocket_scan_score" if selector_mode == "geom_kdtree" else "scan_score",
                            float("-inf"),
                        )
                    ),
                }
                for idx, p in enumerate(scan_selected_pockets)
            ]
            tracer.event(
                stage="scan",
                event_type="scan_selection_ranked",
                payload={"selector_mode": selector_mode, "ranking": ranking_payload},
                level="TRACE",
                decision=True,
            )
            tracer.event(
                stage="scan",
                event_type="scan_selected",
                payload={
                    "selector_mode": selector_mode,
                    "selected_pockets": [str(p.id) for p in scan_selected_pockets],
                },
                level="TRACE",
                decision=True,
            )
            debug_logger.log(
                {
                    "type": "scan_selected",
                    "selector_mode": selector_mode,
                    "selected": [str(p.id) for p in scan_selected_pockets],
                }
            )
        scan_time = time.perf_counter() - scan_start

        if search_space_mode == "reduced" and bool(getattr(cfg, "usar_reducao_condicionada_ao_peptideo", False)):
            tracer.event(
                stage="pocket_filter",
                event_type="reducao_condicionada_ativada",
                payload={"mensagem": "Aplicando subpipeline opcional de reducao guiada por peptideo."},
                level="BASIC",
            )
            scan_selected_pockets = self._aplicar_reducao_condicionada(
                peptide=peptide,
                pockets=scan_selected_pockets,
                cfg=cfg,
                tracer=tracer,
                debug_logger=debug_logger,
            )

        accepted_pockets: list[tuple[int, Pocket]] = [(idx, pocket) for idx, pocket in enumerate(scan_selected_pockets)]
        rejected: list[dict[str, Any]] = []
        if search_space_mode == "reduced" and not modo_legado_pockets:
            accepted_pockets = []
            max_clash_ratio = float(self._obter_valor_cfg(scan_cfg, "max_clash_ratio", 0.02))
            reject_if_feasible_fraction_leq = float(self._obter_valor_cfg(scan_cfg, "reject_if_feasible_fraction_leq", 0.0))
            reject_if_severe_clash_fraction_geq = float(self._obter_valor_cfg(scan_cfg, "reject_if_severe_clash_fraction_geq", 0.95))
            for idx, pocket in enumerate(scan_selected_pockets):
                metrics = scan_results.get(str(pocket.id), {})
                feasible_fraction = float(metrics.get("feasible_fraction", 1.0))
                if selector_mode == "geom_kdtree":
                    severe_clash_fraction = float(metrics.get("severe_clash_fraction", 0.0))
                    if (
                        feasible_fraction <= reject_if_feasible_fraction_leq
                        and severe_clash_fraction >= reject_if_severe_clash_fraction_geq
                    ):
                        reason = "geom_kdtree_clearly_inviable"
                        rejected.append({"pocket_id": pocket.id, "reason": reason})
                        tracer.event(
                            stage="pocket_filter",
                            event_type="pocket_rejected",
                            payload={
                                "selector_mode": selector_mode,
                                "reason": reason,
                                "feasible_fraction": feasible_fraction,
                                "severe_clash_fraction": severe_clash_fraction,
                                "reject_if_feasible_fraction_leq": reject_if_feasible_fraction_leq,
                                "reject_if_severe_clash_fraction_geq": reject_if_severe_clash_fraction_geq,
                                "pocket_scan_score": float(metrics.get("pocket_scan_score", float("-inf"))),
                            },
                            pocket_id=str(pocket.id),
                            level="TRACE",
                            decision=True,
                        )
                        continue
                else:
                    clash_ratio_best = float(metrics.get("clash_ratio_best", 0.0))
                    if feasible_fraction <= 0.0:
                        rejected.append({"pocket_id": pocket.id, "reason": "feasible_fraction<=0.0"})
                        tracer.event(
                            stage="pocket_filter",
                            event_type="pocket_rejected",
                            payload={
                                "selector_mode": selector_mode,
                                "reason": "feasible_fraction<=0.0",
                                "feasible_fraction": feasible_fraction,
                            },
                            pocket_id=str(pocket.id),
                            level="TRACE",
                            decision=True,
                        )
                        continue
                    if clash_ratio_best > max_clash_ratio:
                        rejected.append({"pocket_id": pocket.id, "reason": "clash_ratio_best>max_clash_ratio"})
                        tracer.event(
                            stage="pocket_filter",
                            event_type="pocket_rejected",
                            payload={
                                "selector_mode": selector_mode,
                                "reason": "clash_ratio_best>max_clash_ratio",
                                "clash_ratio_best": clash_ratio_best,
                                "max_clash_ratio": max_clash_ratio,
                            },
                            pocket_id=str(pocket.id),
                            level="TRACE",
                            decision=True,
                        )
                        continue
                tracer.event(
                    stage="pocket_filter",
                    event_type="pocket_metrics_evaluated",
                    payload={"selector_mode": selector_mode, "metrics": metrics, "accepted": True},
                    pocket_id=str(pocket.id),
                    level="TRACE",
                )
                accepted_pockets.append((idx, pocket))

        tracer.event(
            stage="pocket_filter",
            event_type="pocket_acceptance_summary",
            payload={
                "n_candidates": int(len(candidate_pockets)),
                "n_selected_pre_filter": int(len(scan_selected_pockets)),
                "n_accepted": int(len(accepted_pockets)),
                "accepted_pockets": [str(p.id) for _, p in accepted_pockets],
                "rejected_pockets": rejected,
            },
            level="BASIC",
            decision=True,
        )
        debug_logger.log(
            {
                "type": "pocket_acceptance_summary",
                "n_candidates": int(len(candidate_pockets)),
                "n_selected_pre_filter": int(len(scan_selected_pockets)),
                "n_accepted": int(len(accepted_pockets)),
                "accepted": [str(p.id) for _, p in accepted_pockets],
                "rejected": rejected,
            }
        )

        return ContextoSelecaoBolsoes(
            search_space_mode=search_space_mode,
            modo_legado_pockets=modo_legado_pockets,
            candidate_pockets=candidate_pockets,
            selected_pockets=scan_selected_pockets,
            pockets=scan_selected_pockets,
            total_pockets=total_pockets,
            scan_results=scan_results,
            scan_params=scan_params,
            pocketing_time=pocketing_time,
            scan_time=scan_time,
            accepted_pockets=accepted_pockets,
            feasible_pockets=accepted_pockets,
            rejected=rejected,
        )
