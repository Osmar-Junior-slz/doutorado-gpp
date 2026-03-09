"""Orquestrador principal do pipeline de docking."""

from __future__ import annotations

import os
import uuid

import numpy as np

from dockingpp.pipeline import run as run_mod
from dockingpp.pipeline.execucao.auditoria_execucao import AuditoriaExecucaoPipeline
from dockingpp.pipeline.execucao.execucao_busca import ContextoExecucaoBusca, ExecutorBuscaPipeline
from dockingpp.pipeline.execucao.selecao_bolsoes import SelecionadorBolsoesPipeline


class OrquestradorPipelineDocking:
    """Coordena a execução completa mantendo compatibilidade de interface."""

    def executar(self, cfg: run_mod.Config, receptor_path: str, peptide_path: str, out_dir: str) -> run_mod.RunResult:
        """Executa setup, seleção, busca e finalização do pipeline."""

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        np.random.seed(cfg.seed)
        os.makedirs(out_dir, exist_ok=True)

        auditoria = AuditoriaExecucaoPipeline()
        debug_logger = auditoria.criar_logger_debug(out_dir=out_dir, run_id=run_id, cfg=cfg)
        tracer = auditoria.criar_tracer(out_dir=out_dir, run_id=run_id, cfg=cfg)
        auditoria.iniciar_execucao(
            tracer=tracer,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            out_dir=out_dir,
            cfg=cfg,
        )

        try:
            receptor, peptide, dummy_pockets = self._carregar_entradas(receptor_path, peptide_path)
            auditoria.registrar_inputs_carregados(
                tracer=tracer,
                receptor=receptor,
                peptide=peptide,
                extrair_coords=run_mod._extract_coords,
            )

            selecionador = SelecionadorBolsoesPipeline(
                normalizar_modo_busca=run_mod._normalize_search_space_mode,
                obter_valor_cfg=run_mod._cfg_value,
                extrair_coords=run_mod._extract_coords,
                aplicar_reducao_condicionada=run_mod._aplicar_reducao_condicionada_ao_peptideo,
            )
            contexto_selecao = selecionador.selecionar(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                dummy_pockets=dummy_pockets,
                tracer=tracer,
                debug_logger=debug_logger,
            )

            executor = ExecutorBuscaPipeline(
                executar_execucao_unica=run_mod._execute_single_run,
                escrever_debug_summary=auditoria.escrever_debug_summary,
                alocar_orcamento=run_mod._allocate_split_budget,
                construir_bolsao_global=run_mod._build_global_pocket,
                classe_config=run_mod.Config,
            )
            return executor.executar(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                out_dir=out_dir,
                run_id=run_id,
                tracer=tracer,
                contexto=ContextoExecucaoBusca(
                    search_space_mode=contexto_selecao.search_space_mode,
                    modo_legado_pockets=contexto_selecao.modo_legado_pockets,
                    candidate_pockets=contexto_selecao.candidate_pockets,
                    selected_pockets=contexto_selecao.selected_pockets,
                    pockets=contexto_selecao.pockets,
                    total_pockets=contexto_selecao.total_pockets,
                    scan_results=contexto_selecao.scan_results,
                    scan_params=contexto_selecao.scan_params,
                    pocketing_time=contexto_selecao.pocketing_time,
                    scan_time=contexto_selecao.scan_time,
                    accepted_pockets=contexto_selecao.accepted_pockets,
                    feasible_pockets=contexto_selecao.feasible_pockets,
                    rejected=contexto_selecao.rejected,
                ),
            )
        except Exception as exc:
            auditoria.registrar_falha_execucao(tracer=tracer, erro=exc)
            raise
        finally:
            auditoria.finalizar_execucao(tracer=tracer, debug_logger=debug_logger, cfg=cfg, out_dir=out_dir)

    @staticmethod
    def _carregar_entradas(receptor_path: str, peptide_path: str):
        """Carrega entradas reais ou dados dummy para testes."""

        if receptor_path == "__dummy__" and peptide_path == "__dummy__":
            receptor, peptide, dummy_pockets = run_mod._dummy_inputs()
        else:
            receptor = run_mod.load_receptor(receptor_path)
            peptide = run_mod.load_peptide(peptide_path)
            dummy_pockets = []
        return receptor, peptide, dummy_pockets
