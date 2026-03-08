"""Pipeline entrypoints."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field

from dockingpp.data.io import load_peptide, load_pockets, load_receptor
from dockingpp.data.structs import Pocket, RunResult
from dockingpp.pipeline.logging import AuditTracer, RunLogger
from dockingpp.pipeline.execucao.auditoria_execucao import AuditoriaExecucaoPipeline
from dockingpp.pipeline.execucao.agregacao_resultados import AgregadorResultadosPipeline
from dockingpp.pipeline.scan import (
    build_receptor_kdtree,
    scan_pocket_feasibility_geom_kdtree,
    scan_pocket_feasibility,
    select_pockets_from_scan,
)
from dockingpp.priors.pocket import PriorNetPocket, rank_pockets
from dockingpp.priors.pose import PriorNetPose
from dockingpp.reducao.admissibilidade import avaliar_admissibilidade_bolsao
from dockingpp.reducao.geometria_bolsao import descrever_geometria_bolsao
from dockingpp.reducao.perfil_peptideo import construir_perfil_peptideo
from dockingpp.reducao.pre_afinidade import estimar_pre_afinidade_bolsao
from dockingpp.reducao.ranking_bolsoes import ranquear_bolsoes_candidatos
from dockingpp.reducao.seletor_bolsoes import selecionar_bolsoes_para_busca
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
    usar_reducao_condicionada_ao_peptideo: bool = False

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
    peptide_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    pockets = [
        Pocket(id="dummy-0", center=np.array([0.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords),
        Pocket(id="dummy-1", center=np.array([10.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords),
        Pocket(id="dummy-2", center=np.array([0.0, 10.0, 0.0]), radius=5.0, coords=receptor_coords),
    ]
    receptor = {"dummy": True, "coords": receptor_coords}
    peptide = {"dummy": True, "coords": peptide_coords}
    return receptor, peptide, pockets


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


def _aplicar_reducao_condicionada_ao_peptideo(
    peptide: Any,
    pockets: list[Pocket],
    cfg: Config,
    tracer: AuditTracer,
    debug_logger: DebugLogger,
) -> list[Pocket]:
    """Aplica subpipeline opcional de redução guiada por peptídeo."""

    if not pockets:
        return pockets

    cfg_reducao = _cfg_value(cfg, "reducao", {}) or {}
    if not isinstance(cfg_reducao, dict):
        cfg_reducao = {}

    top_k_reducao = cfg_reducao.get("top_k")
    if top_k_reducao is None:
        top_k_reducao = len(pockets)

    score_minimo = cfg_reducao.get("score_minimo")
    apenas_admissiveis = bool(cfg_reducao.get("apenas_admissiveis", True))
    permitir_fallback = bool(cfg_reducao.get("permitir_fallback", True))
    quantidade_fallback = int(cfg_reducao.get("quantidade_fallback", 1) or 1)

    perfil_peptideo = construir_perfil_peptideo(peptide)
    debug_logger.log(
        {
            "type": "reducao_perfil_peptideo",
            "mensagem": "Perfil geométrico do peptídeo para redução condicionada.",
            "comprimento_efetivo": float(perfil_peptideo.comprimento_efetivo),
            "largura_efetiva": float(perfil_peptideo.largura_efetiva),
            "espessura_efetiva": float(perfil_peptideo.espessura_efetiva),
            "extensao_maxima": float(perfil_peptideo.extensao_maxima),
            "raio_giro": float(perfil_peptideo.raio_giro),
            "indice_flexibilidade": float(perfil_peptideo.indice_flexibilidade),
        }
    )

    geometrias_por_id = {str(pocket.id): descrever_geometria_bolsao(pocket) for pocket in pockets}
    resultados_admissibilidade = [
        avaliar_admissibilidade_bolsao(perfil_peptideo, geometrias_por_id[str(pocket.id)])
        for pocket in pockets
    ]
    for resultado in resultados_admissibilidade:
        geometria = geometrias_por_id.get(resultado.id_bolsao)
        debug_logger.log(
            {
                "type": "reducao_bolsao_avaliado",
                "mensagem": "Avaliação geométrica de admissibilidade do bolsão.",
                "id_bolsao": resultado.id_bolsao,
                "admissivel": bool(resultado.admissivel),
                "score_encaixe_geometrico": float(resultado.score_encaixe_geometrico),
                "motivos_reprovacao": list(resultado.motivos_reprovacao),
                "geometria": {
                    "comprimento_util": float(geometria.comprimento_util) if geometria is not None else None,
                    "largura_util": float(geometria.largura_util) if geometria is not None else None,
                    "profundidade_util": float(geometria.profundidade_util) if geometria is not None else None,
                    "continuidade_superficial": float(geometria.continuidade_superficial) if geometria is not None else None,
                    "exposicao_superficial": float(geometria.exposicao_superficial) if geometria is not None else None,
                    "volume_estimado": float(geometria.volume_estimado) if geometria is not None else None,
                },
            }
        )
        tracer.event(
            stage="pocket_filter",
            event_type="reducao_bolsao_avaliado",
            payload={
                "id_bolsao": resultado.id_bolsao,
                "admissivel": bool(resultado.admissivel),
                "score_encaixe_geometrico": float(resultado.score_encaixe_geometrico),
                "motivos_reprovacao": list(resultado.motivos_reprovacao),
            },
            pocket_id=str(resultado.id_bolsao),
            level="TRACE",
            decision=True,
        )

    resultados_pre_afinidade = [
        estimar_pre_afinidade_bolsao(
            perfil_peptideo,
            geometrias_por_id[resultado.id_bolsao],
            score_encaixe_geometrico=resultado.score_encaixe_geometrico,
        )
        for resultado in resultados_admissibilidade
        if resultado.admissivel
    ]
    for resultado_pre in resultados_pre_afinidade:
        debug_logger.log(
            {
                "type": "reducao_pre_afinidade",
                "mensagem": "Pré-afinidade barata calculada para bolsão admissível.",
                "id_bolsao": resultado_pre.id_bolsao,
                "score_pre_afinidade": float(resultado_pre.score_pre_afinidade),
                "score_contatos": float(resultado_pre.score_contatos),
                "penalidade_clash": float(resultado_pre.penalidade_clash),
                "score_ancoragem": float(resultado_pre.score_ancoragem),
            }
        )

    ranking_reducao = ranquear_bolsoes_candidatos(resultados_admissibilidade, resultados_pre_afinidade)
    for posicao, entrada in enumerate(ranking_reducao, start=1):
        debug_logger.log(
            {
                "type": "reducao_ranking",
                "mensagem": "Posição do bolsão no ranking da redução.",
                "id_bolsao": entrada.id_bolsao,
                "posicao": int(posicao),
                "admissivel": bool(entrada.admissivel),
                "score_encaixe_geometrico": float(entrada.score_encaixe_geometrico),
                "score_pre_afinidade": float(entrada.score_pre_afinidade),
                "score_final": float(entrada.score_final),
            }
        )

    selecionados_ranking = selecionar_bolsoes_para_busca(
        ranking_reducao,
        top_k=top_k_reducao,
        score_minimo=score_minimo,
        apenas_admissiveis=apenas_admissiveis,
        permitir_fallback=permitir_fallback,
        quantidade_fallback=quantidade_fallback,
    )

    mapa_pockets = {str(pocket.id): pocket for pocket in pockets}
    pockets_selecionados = [
        mapa_pockets[item.id_bolsao]
        for item in selecionados_ranking
        if item.id_bolsao in mapa_pockets
    ]

    fallback_acionado = (not selecionados_ranking) and bool(permitir_fallback) and bool(ranking_reducao)

    tracer.event(
        stage="pocket_filter",
        event_type="reducao_condicionada_aplicada",
        payload={
            "n_pockets_entrada": int(len(pockets)),
            "n_pockets_saida": int(len(pockets_selecionados)),
            "top_k": int(top_k_reducao) if top_k_reducao is not None else None,
            "score_minimo": float(score_minimo) if score_minimo is not None else None,
            "apenas_admissiveis": bool(apenas_admissiveis),
            "permitir_fallback": bool(permitir_fallback),
            "quantidade_fallback": int(quantidade_fallback),
            "fallback_acionado": bool(fallback_acionado),
            "selecionados": [item.id_bolsao for item in selecionados_ranking],
            "n_pre_afinidade": int(len(resultados_pre_afinidade)),
        },
        level="BASIC",
        decision=True,
    )
    debug_logger.log(
        {
            "type": "reducao_condicionada",
            "mensagem": "Subpipeline de redução guiada por peptídeo ativado.",
            "n_pockets_entrada": int(len(pockets)),
            "n_pockets_saida": int(len(pockets_selecionados)),
            "regras_selecao": {
                "top_k": int(top_k_reducao) if top_k_reducao is not None else None,
                "score_minimo": float(score_minimo) if score_minimo is not None else None,
                "apenas_admissiveis": bool(apenas_admissiveis),
                "permitir_fallback": bool(permitir_fallback),
                "quantidade_fallback": int(quantidade_fallback),
            },
            "selecionados": [item.id_bolsao for item in selecionados_ranking],
            "fallback_acionado": bool(fallback_acionado),
            "n_pre_afinidade": int(len(resultados_pre_afinidade)),
        }
    )

    return pockets_selecionados


def _normalize_search_space_mode(search_space_mode: str | None, full_search: bool) -> str:
    raw = (search_space_mode or "").strip().lower()
    if raw in {"full", "reduced"}:
        return raw
    if raw == "global":
        return "full"
    if raw == "pockets":
        # Compatibilidade: alias legado "pockets" seguia `full_search`.
        return "full" if full_search else "reduced"
    return "full" if full_search else "reduced"


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
    scan_results: dict[str, dict[str, Any]],
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

    pockets_para_busca = pockets
    # Regra histórica: no modo full, `max_pockets_used` limita o conjunto
    # entregue ao motor de busca sem alterar a seleção upstream.
    if search_space_mode == "full":
        limite_pockets = int(getattr(cfg, "max_pockets_used", 0) or 0)
        if limite_pockets > 0:
            pockets_para_busca = pockets[:limite_pockets]

    auditoria = AuditoriaExecucaoPipeline()
    auditoria.registrar_inicio_busca(tracer=tracer, cfg=cfg, pocket_id=pocket_id)
    start_search = time.perf_counter()
    result = search.search(
        receptor=receptor,
        peptide=peptide,
        pockets=pockets_para_busca,
        cfg=cfg,
        score_cheap_fn=score_pose_cheap,
        score_expensive_fn=score_pose_expensive,
        prior_pocket=prior_pocket,
        prior_pose=prior_pose,
        logger=logger,
    )
    end_search = time.perf_counter()
    auditoria.registrar_fim_busca(tracer=tracer, pocket_id=pocket_id, runtime_sec=end_search - start_search)

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

    agregador = AgregadorResultadosPipeline()
    payload = agregador.construir_payload_execucao(
        run_id=run_id,
        mode="single",
        search_space_mode=search_space_mode,
        runtime_sec=runtime_sec,
        total_pockets=total_pockets,
        selected_pockets=len(pockets_para_busca),
        best_score_cheap=result.best_pose.score_cheap,
        best_score_expensive=result.best_pose.score_expensive,
        best_pose_id=best_pose_id,
        config_resolved_subset=config_resolved_subset,
        pocketing_time=pocketing_time,
        scan_time=scan_time,
        search_time=end_search - start_search,
    )
    result_path = agregador.persistir_payload_resultado(out_dir=out_dir, payload=payload)
    auditoria.registrar_artefato_escrito(tracer=tracer, caminho=result_path)

    logger.flush(out_dir)
    logger.flush_timeseries(out_dir, mode=search_space_mode)
    summary_path = agregador.escrever_summary_execucao(
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
        selected_pocket_ids=[str(p.id) for p in pockets_para_busca],
    )
    auditoria.registrar_artefato_escrito(tracer=tracer, caminho=summary_path)
    auditoria.registrar_execucao_finalizada(
        tracer=tracer,
        pocket_id=pocket_id,
        best_score_cheap=result.best_pose.score_cheap,
        best_score_expensive=result.best_pose.score_expensive,
    )
    return result, payload, logger


def _executar_pipeline_fluxo(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Compatibilidade: delega o fluxo principal ao orquestrador dedicado."""

    from dockingpp.pipeline.execucao.orquestrador import OrquestradorPipelineDocking

    return OrquestradorPipelineDocking().executar(
        cfg=cfg,
        receptor_path=receptor_path,
        peptide_path=peptide_path,
        out_dir=out_dir,
    )



def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Fachada pública de execução do pipeline.

    A orquestração principal foi movida para `pipeline.execucao` para reduzir
    acoplamento e deixar este módulo mais enxuto como ponto de entrada.
    """

    from dockingpp.pipeline.execucao.orquestrador import OrquestradorPipelineDocking

    return OrquestradorPipelineDocking().executar(
        cfg=cfg,
        receptor_path=receptor_path,
        peptide_path=peptide_path,
        out_dir=out_dir,
    )


