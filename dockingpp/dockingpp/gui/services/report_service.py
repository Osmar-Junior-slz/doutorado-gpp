"""Serviços utilitários para carregar relatórios e métricas de execução."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

JsonKind = Literal["compare", "single", "unknown"]


@dataclass(frozen=True)
class ReportBundle:
    """Agrupa os dados do relatório carregados de pasta ou upload."""

    kind: JsonKind
    main_json: dict[str, Any]
    metrics: list[dict[str, Any]] | None
    aux_jsons: dict[str, dict[str, Any]]


def find_runs(root_dir: Path) -> list[Path]:
    """Encontra pastas de execução contendo pelo menos um arquivo JSON/JSONL."""

    if not root_dir.exists():
        return []
    # PT-BR: filtramos somente diretórios com arquivos de saída relevantes.
    runs = [
        child
        for child in root_dir.iterdir()
        if child.is_dir() and (any(child.glob("*.json")) or any(child.glob("*.jsonl")))
    ]
    return sorted(runs)


def load_json(path: Path) -> dict[str, Any]:
    """Carrega um arquivo JSON do disco."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Carrega um arquivo JSONL do disco."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            # PT-BR: cada linha representa um registro independente em JSON.
            records.append(json.loads(line))
    return records


def infer_json_kind(obj: dict[str, Any]) -> JsonKind:
    """Infere o tipo do relatório com base em chaves conhecidas."""

    if "full" in obj and "reduced" in obj:
        return "compare"
    runs = obj.get("runs")
    if isinstance(runs, dict) and "full" in runs and "reduced" in runs:
        return "compare"
    comparison = obj.get("comparison")
    if isinstance(comparison, dict) and "full" in comparison and "reduced" in comparison:
        return "compare"
    if obj.get("mode") == "compare":
        return "compare"

    if any(key in obj for key in ("best_score_cheap", "best_score")):
        return "single"
    if any(key in obj for key in ("n_eval", "evaluations")):
        return "single"
    if any(key in obj for key in ("pose", "best_pose")):
        return "single"
    if any(key in obj for key in ("config", "cfg")):
        return "single"

    return "unknown"


def summarize_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Resume métricas em um dicionário nome -> valor."""

    summary: dict[str, Any] = {}
    for record in records:
        name = record.get("name")
        if not name:
            continue
        # PT-BR: preferimos a última ocorrência do mesmo nome.
        summary[name] = record.get("value")
    return summary


def _aggregate_series_min(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrega uma série por passo, mantendo o menor valor em cada step."""

    min_by_step: dict[int, float] = {}
    for item in series:
        step = item.get("step")
        score = item.get("score")
        if step is None or score is None:
            continue
        step_int = int(step)
        score_val = float(score)
        # PT-BR: usamos o menor score por step para eliminar duplicatas serrilhadas.
        if step_int not in min_by_step or score_val < min_by_step[step_int]:
            min_by_step[step_int] = score_val
    return [{"step": step, "score": min_by_step[step]} for step in sorted(min_by_step)]


def _apply_cumulative_min(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aplica o melhor-so-far por mínimo cumulativo, garantindo monotonicidade."""

    best_so_far: float | None = None
    cumulative: list[dict[str, Any]] = []
    for item in series:
        score = item.get("score")
        if score is None:
            continue
        score_val = float(score)
        if best_so_far is None or score_val < best_so_far:
            best_so_far = score_val
        # PT-BR: garantimos séries não-crescentes (melhor = menor score).
        cumulative.append({"step": int(item["step"]), "score": best_so_far})
    return cumulative


def metrics_series(
    records: list[dict[str, Any]],
    keys: list[str],
    step_keys: list[str] | None = None,
    aggregate: Literal["min"] | None = None,
    cumulative_best: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Monta uma série para gráficos a partir de registros de métricas."""

    if not records:
        return [], None
    if step_keys is None:
        # PT-BR: priorizamos "step" global para evitar duplicatas por pocket.
        step_keys = ["step", "generation", "gen", "iter"]

    selected_key = next((key for key in keys if any(rec.get("name") == key for rec in records)), None)
    series: list[dict[str, Any]] = []
    if selected_key:
        for idx, record in enumerate(records):
            if record.get("name") != selected_key:
                continue
            step = next((record.get(key) for key in step_keys if record.get(key) is not None), None)
            if step is None:
                step = idx
            series.append({"step": step, "score": record.get("value")})
        if aggregate == "min":
            series = _aggregate_series_min(series)
        if cumulative_best:
            series = _apply_cumulative_min(series)
        return series, selected_key

    selected_key = next((key for key in keys if any(key in rec for rec in records)), None)
    if not selected_key:
        return [], None

    for idx, record in enumerate(records):
        if selected_key not in record:
            continue
        step = next((record.get(key) for key in step_keys if record.get(key) is not None), None)
        if step is None:
            step = idx
        series.append({"step": step, "score": record.get(selected_key)})
    if aggregate == "min":
        series = _aggregate_series_min(series)
    if cumulative_best:
        series = _apply_cumulative_min(series)
    return series, selected_key


def _extract_compare_block(report_data: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Busca o bloco de comparação correspondente ao rótulo fornecido."""

    # PT-BR: tentamos diferentes formatos de relatório aceitos historicamente.
    for container in (report_data, report_data.get("runs"), report_data.get("comparison")):
        if isinstance(container, dict) and label in container and isinstance(container[label], dict):
            return container[label]
    return None


def _lookup_metric(block: dict[str, Any], keys: Iterable[str]) -> Any:
    """Procura um valor de métrica em um bloco, considerando chaves alternativas."""

    for key in keys:
        if key in block and block[key] is not None:
            return block[key]
    metrics = block.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]
    return None


def build_compare_table(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Monta a tabela de comparação entre modos completo e reduzido."""

    rows: list[dict[str, Any]] = []
    for label in ("full", "reduced"):
        block = _extract_compare_block(report_data, label)
        if not block:
            continue
        label_name = "Completo" if label == "full" else "Reduzido"
        # PT-BR: usamos chaves alternativas para compatibilidade retroativa.
        rows.append(
            {
                "Modo": label_name,
                "Melhor score (cheap)": _lookup_metric(block, ["best_score_cheap", "best_score", "best"]),
                "Avaliações": _lookup_metric(block, ["n_eval", "evals", "evaluations"]),
                "Bolsões totais": _lookup_metric(block, ["n_pockets_total", "pockets_total"]),
                "Bolsões usados": _lookup_metric(block, ["n_pockets_used", "pockets_used"]),
                "Razão de redução": _lookup_metric(block, ["reduction_ratio", "ratio"]),
                "Tempo (s)": _lookup_metric(block, ["elapsed_s", "elapsed_seconds", "elapsed"]),
            }
        )
    return rows
