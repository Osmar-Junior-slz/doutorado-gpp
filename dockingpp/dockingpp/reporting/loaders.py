"""Carregadores tolerantes para relatórios (PT-BR)."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Literal


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

RunKind = Literal["full", "reduced", "unknown"]


@dataclass(frozen=True)
class ReportRun:
    """Describe os arquivos associados a uma execução."""

    run_dir: Path
    kind: RunKind
    summary_path: Path | None
    result_path: Path | None
    config_path: Path | None
    metrics_path: Path | None
    metrics_timeseries_path: Path | None
    run_id: str | None = None

    def label(self) -> str:
        """Rótulo humano para seleção na UI."""

        return self.run_id or self.run_dir.name


def find_report_runs(root_dir: Path) -> list[ReportRun]:
    """Varre uma pasta e encontra execuções single (Full/Reduced) com base no conteúdo."""

    if not root_dir.exists():
        return []

    json_paths = list(root_dir.rglob("*.json"))
    if not json_paths:
        return []

    dirs = {path.parent for path in json_paths}
    if any(path.parent == root_dir for path in json_paths):
        dirs.add(root_dir)

    runs: list[ReportRun] = []
    for run_dir in sorted(dirs):
        run = _build_report_run(run_dir)
        if run is not None:
            runs.append(run)
    return runs


def pair_full_reduced(runs: Iterable[ReportRun]) -> tuple[ReportRun | None, ReportRun | None]:
    """Seleciona um par Full/Reduced entre as execuções encontradas."""

    full = None
    reduced = None
    for run in sorted(runs, key=lambda item: item.label()):
        if run.kind == "full" and full is None:
            full = run
        elif run.kind == "reduced" and reduced is None:
            reduced = run
    return full, reduced


def _build_report_run(run_dir: Path) -> ReportRun | None:
    json_paths = sorted(run_dir.glob("*.json"))
    if not json_paths:
        return None

    summary_path = None
    result_path = None
    config_path = None
    payloads: list[dict[str, Any]] = []
    run_id = None

    for path in json_paths:
        try:
            payload = load_any_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        payloads.append(payload)
        role = _classify_json_payload(payload)
        if role == "summary":
            summary_path = path
        elif role == "result":
            result_path = path
            run_id = _safe_str(payload.get("run_id")) or run_id
        elif role == "config":
            config_path = path

    kind = _infer_run_kind(payloads)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_timeseries_path = run_dir / "metrics.timeseries.jsonl"
    return ReportRun(
        run_dir=run_dir,
        kind=kind,
        summary_path=summary_path,
        result_path=result_path,
        config_path=config_path,
        metrics_path=metrics_path if metrics_path.exists() else None,
        metrics_timeseries_path=metrics_timeseries_path if metrics_timeseries_path.exists() else None,
        run_id=run_id,
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _classify_json_payload(payload: dict[str, Any]) -> str:
    if _is_summary_payload(payload):
        return "summary"
    if _is_result_payload(payload):
        return "result"
    if _is_config_payload(payload):
        return "config"
    return "unknown"


def _is_summary_payload(payload: dict[str, Any]) -> bool:
    return bool(
        isinstance(payload.get("timing"), dict)
        and payload.get("best_score_cheap") is not None
        and payload.get("mode") is not None
    )


def _is_result_payload(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("run_id") is not None
        and payload.get("best_cheap_by_pocket") is not None
        and payload.get("n_eval_total") is not None
    )


def _is_config_payload(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("seed") is not None
        and payload.get("generations") is not None
        and payload.get("pop_size") is not None
        and (_extract_flag(payload, "full_search") is not None or _extract_flag(payload, "search_space_mode") is not None)
    )


def _extract_flag(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    for container_key in ("config", "cfg", "settings", "options"):
        nested = payload.get(container_key)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)
    return None


def _infer_run_kind(payloads: Iterable[dict[str, Any]]) -> RunKind:
    full_signal = False
    reduced_signal = False
    for payload in payloads:
        full_search = _extract_flag(payload, "full_search")
        if full_search is True:
            full_signal = True
        elif full_search is False:
            reduced_signal = True
        search_space_mode = _extract_flag(payload, "search_space_mode")
        if search_space_mode == "global":
            full_signal = True
        elif search_space_mode == "pockets":
            reduced_signal = True
    if full_signal:
        return "full"
    if reduced_signal:
        return "reduced"
    return "unknown"


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
