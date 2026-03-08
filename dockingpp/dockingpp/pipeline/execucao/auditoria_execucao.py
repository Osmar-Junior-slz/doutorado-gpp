"""Serviços de auditoria, tracing e debug da execução do pipeline."""

from __future__ import annotations

from typing import Any, Callable

from dockingpp.pipeline.logging import AuditTracer
from dockingpp.utils.debug_logger import DebugLogger


class AuditoriaExecucaoPipeline:
    """Centraliza emissão de eventos estruturados e controle de tracing.

    A intenção é tirar de `run.py` os detalhes de auditoria/debug,
    mantendo formatos de eventos e artefatos já validados.
    """

    def criar_logger_debug(self, *, out_dir: str, run_id: str, cfg: Any) -> DebugLogger:
        """Cria e associa `DebugLogger` à configuração da execução."""

        debug_log_enabled = bool(getattr(cfg, "debug_log_enabled", False))
        debug_log_path = getattr(cfg, "debug_log_path", None) or f"{out_dir}/debug/debug.jsonl"
        debug_log_level = str(getattr(cfg, "debug_log_level", "INFO"))
        debug_logger = DebugLogger(enabled=debug_log_enabled, path=debug_log_path, level=debug_log_level)
        debug_logger.run_id = run_id
        cfg.debug_logger = debug_logger
        return debug_logger

    def criar_tracer(self, *, out_dir: str, run_id: str, cfg: Any) -> AuditTracer:
        """Cria e associa `AuditTracer` à configuração da execução."""

        tracer = AuditTracer(
            out_dir=out_dir,
            run_id=run_id,
            debug_enabled=bool(getattr(cfg, "debug_enabled", True)),
            debug_level=str(getattr(cfg, "debug_level", "AUDIT")),
            debug_dirname=str(getattr(cfg, "debug_dirname", "debug")),
            search_space_mode=str(getattr(cfg, "search_space_mode", "full")),
        )
        cfg.audit_tracer = tracer
        return tracer

    def iniciar_execucao(self, *, tracer: AuditTracer, receptor_path: str, peptide_path: str, out_dir: str, cfg: Any) -> None:
        """Registra evento inicial do ciclo de execução."""

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

    def registrar_inputs_carregados(self, *, tracer: AuditTracer, receptor: Any, peptide: Any, extrair_coords: Callable[[Any], Any]) -> None:
        """Registra metadados dos inputs carregados para o pipeline."""

        tracer.event(
            stage="io",
            event_type="inputs_loaded",
            payload={
                "receptor_atoms": int(extrair_coords(receptor).shape[0]),
                "peptide_atoms": int(extrair_coords(peptide).shape[0]),
            },
            level="BASIC",
        )

    def registrar_inicio_busca(self, *, tracer: AuditTracer | None, cfg: Any, pocket_id: str | None) -> None:
        """Registra início da etapa de busca do motor."""

        if tracer is None:
            return
        tracer.event(
            stage="search",
            substage="start",
            event_type="search_started",
            payload={
                "pocket_id": pocket_id,
                "engine_name": "ABCGAVGOSSearch",
                "generations": int(cfg.generations),
                "pop_size": int(cfg.pop_size),
            },
            engine="ABCGAVGOSSearch",
            pocket_id=pocket_id,
            level="BASIC",
        )

    def registrar_fim_busca(self, *, tracer: AuditTracer | None, pocket_id: str | None, runtime_sec: float) -> None:
        """Registra término da etapa de busca."""

        if tracer is None:
            return
        tracer.event(
            stage="search",
            substage="end",
            event_type="generation_completed",
            payload={"pocket_id": pocket_id, "runtime_sec": float(runtime_sec)},
            engine="ABCGAVGOSSearch",
            pocket_id=pocket_id,
            level="TRACE",
        )

    def registrar_artefato_escrito(self, *, tracer: AuditTracer | None, caminho: str) -> None:
        """Registra artefatos persistidos pela execução."""

        if tracer is not None:
            tracer.artifact_written(caminho)

    def registrar_execucao_finalizada(self, *, tracer: AuditTracer | None, pocket_id: str | None, best_score_cheap: float | None, best_score_expensive: float | None) -> None:
        """Registra evento final de sucesso da execução."""

        if tracer is None:
            return
        tracer.event(
            stage="summary",
            event_type="run_finished",
            payload={
                "status": "success",
                "best_score_cheap": best_score_cheap,
                "best_score_expensive": best_score_expensive,
            },
            level="BASIC",
            pocket_id=pocket_id,
        )

    def registrar_falha_execucao(self, *, tracer: AuditTracer, erro: Exception) -> None:
        """Registra falha estruturada da execução."""

        tracer.error(stage="summary", message="pipeline_exception", payload={"error": type(erro).__name__})

    def finalizar_execucao(self, *, tracer: AuditTracer, debug_logger: DebugLogger, cfg: Any, out_dir: str) -> None:
        """Finaliza run, escreve manifesto final e fecha logger de debug."""

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

    def escrever_debug_summary(self, tracer: AuditTracer, payload: dict[str, Any], rel_path: str = "debug_summary.json") -> None:
        """Escreve `debug_summary` no formato de auditoria já existente."""

        if not tracer.enabled:
            return
        data = {
            "schema_version": "1.0",
            "run_id": tracer.run_id,
            **payload,
        }
        tracer.write_summary(data, rel_path=rel_path)
