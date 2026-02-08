"""Carregadores tolerantes para relatórios (PT-BR)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CANDIDATAS_SERIES: dict[str, list[str]] = {
    "iter": ["iter", "generation", "gen", "step"],
    "best_cheap": ["best_score_cheap", "best_cheap", "best"],
    "best_expensive": ["best_score_expensive", "best_expensive", "best_energy", "vina"],
    "n_eval_total": ["n_eval_total", "eval_total", "n_scored", "cheap_evals"],
    "n_filtered": ["n_filtered", "filtered", "discarded"],
    "n_selected": ["n_selected", "selected", "kept"],
    "runtime_s": ["runtime_s", "time_s", "elapsed_s"],
    "expensive_ran": ["expensive_ran", "n_expensive", "expensive_calls"],
    "pocket_id": ["pocket_id", "pocket"],
    "pocket_rank": ["pocket_rank", "rank"],
}


def load_any_json(path: str | Path) -> dict[str, Any]:
    """Carrega qualquer JSON e garante retorno em dicionário."""

    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict):
        return payload
    return {"data": payload}


def find_matching_jsonl(json_path: str | Path) -> Path | None:
    """Localiza um JSONL compatível com base em heurísticas simples."""

    path = Path(json_path)
    candidate = path.with_suffix(".jsonl")
    if candidate.exists():
        return candidate

    try:
        payload = load_any_json(path)
    except (OSError, json.JSONDecodeError):
        payload = {}

    metrics_path = payload.get("metrics_path")
    if metrics_path:
        metrics_candidate = Path(metrics_path).expanduser()
        if not metrics_candidate.is_absolute():
            metrics_candidate = path.parent / metrics_candidate
        if metrics_candidate.exists():
            return metrics_candidate

    out_dir = payload.get("outdir") or payload.get("out_dir") or payload.get("output_dir")
    if out_dir:
        out_path = Path(out_dir).expanduser()
        if not out_path.is_absolute():
            out_path = path.parent / out_path
        metrics_jsonl = out_path / "metrics.jsonl"
        if metrics_jsonl.exists():
            return metrics_jsonl
    return None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Carrega JSONL tolerante, ignorando linhas inválidas."""

    records: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def extract_series(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extrai séries padronizadas de eventos JSONL."""

    series: dict[str, list[Any]] = {key: [] for key in _CANDIDATAS_SERIES}
    faltas: dict[str, int] = {key: 0 for key in _CANDIDATAS_SERIES if key != "iter"}

    for idx, evento in enumerate(events):
        iter_valor = _buscar_valor(evento, _CANDIDATAS_SERIES["iter"])
        if iter_valor is None:
            iter_valor = idx
        series["iter"].append(iter_valor)

        for nome_serie, chaves in _CANDIDATAS_SERIES.items():
            if nome_serie == "iter":
                continue
            valor = _buscar_valor(evento, chaves)
            if valor is None and _tem_nome_valor(evento, chaves):
                valor = evento.get("value")
            if valor is None:
                faltas[nome_serie] += 1
            series[nome_serie].append(valor)

    series["missing"] = faltas
    return series


def _buscar_valor(evento: dict[str, Any], chaves: list[str]) -> Any:
    for chave in chaves:
        if evento.get(chave) is not None:
            return evento.get(chave)
    extras = evento.get("extras")
    if isinstance(extras, dict):
        for chave in chaves:
            if extras.get(chave) is not None:
                return extras.get(chave)
    return None


def _tem_nome_valor(evento: dict[str, Any], chaves: list[str]) -> bool:
    nome = evento.get("name")
    return nome is not None and nome in chaves
