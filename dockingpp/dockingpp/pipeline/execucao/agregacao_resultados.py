"""Agregação e persistência de resultados do pipeline."""

from __future__ import annotations

import json
import os
from typing import Any


class AgregadorResultadosPipeline:
    """Consolida resultados de execução e persiste artefatos finais.

    Esta classe centraliza a montagem de `result.json` e `summary.json`
    sem alterar a semântica dos payloads já validados nos testes.
    """

    def construir_payload_execucao(
        self,
        *,
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
        """Monta payload padrão de execução única."""

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

    def persistir_payload_resultado(self, *, out_dir: str, payload: dict[str, Any]) -> str:
        """Escreve `result.json` e retorna o caminho do artefato."""

        result_path = os.path.join(out_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
        return result_path

    def escrever_summary_execucao(
        self,
        *,
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
        pockets: list[Any],
        scan_params: dict[str, Any],
        scan_by_pocket: dict[str, dict[str, Any]],
        selected_pocket_ids: list[str],
    ) -> str:
        """Consolida métricas finais e escreve `summary.json`."""

        expensive_ran = 0.0
        expensive_skipped = 0.0
        n_eval_total = 0.0
        best_by_pocket: dict[int, float] = {}

        for record in records:
            name = record.get("name")
            value = self._safe_float(record.get("value"))
            if name == "expensive_ran" and value is not None:
                expensive_ran += value
            elif name == "expensive_skipped" and value is not None:
                expensive_skipped += value
            elif name == "n_eval" and value is not None:
                n_eval_total += value
            elif name == "best_score" and value is not None:
                pocket_index = self._record_field(record, "pocket_index")
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
        return summary_path

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Converte para float, retornando `None` para valores inválidos."""

        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _record_field(record: dict[str, Any], key: str) -> Any:
        """Obtém campo de record preservando compatibilidade de formato."""

        if key in record:
            return record[key]
        payload = record.get("payload")
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        return None
