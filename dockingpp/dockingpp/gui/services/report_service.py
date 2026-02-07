"""Serviços utilitários para carregar relatórios e métricas de execução."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

JsonKind = Literal["compare", "single", "unknown"]
CumulativeMode = Literal["min", "max"]

CUMULATIVE_BEST_BY_KEY: dict[str, CumulativeMode] = {
    "best_score": "max",
    "best_score_cheap": "max",
    "best_score_expensive": "max",
    "best": "max",
    "mean_score": "max",
    "n_clashes": "min",
}


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


def load_metrics_series(metrics_jsonl_path: Path | str) -> list[dict[str, Any]]:
    """Carrega métricas de um arquivo JSONL em uma lista de registros."""

    path = Path(metrics_jsonl_path)
    return load_jsonl(path)


def best_so_far(series: list[dict[str, Any]], mode: CumulativeMode = "min") -> list[dict[str, Any]]:
    """Aplica melhor-so-far (cumulativo) em uma série de scores."""

    best_value: float | None = None
    cumulative: list[dict[str, Any]] = []
    for item in sorted(series, key=lambda entry: entry.get("step", 0)):
        score = item.get("score")
        if score is None:
            continue
        score_val = float(score)
        if best_value is None:
            best_value = score_val
        elif mode == "min" and score_val < best_value:
            best_value = score_val
        elif mode == "max" and score_val > best_value:
            best_value = score_val
        cumulative.append({"step": int(item.get("step", 0)), "score": best_value})
    return cumulative


def aggregate_cost(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrega o custo total (n_eval) acumulado a partir das métricas."""

    if not records:
        return []

    def extract_step(record: dict[str, Any], idx: int) -> int:
        for key in ("step", "generation", "gen", "iter"):
            if record.get(key) is not None:
                return int(record[key])
        return idx

    eval_deltas: list[tuple[int, float]] = []
    for idx, record in enumerate(records):
        if record.get("name") == "n_eval":
            value = record.get("value")
            if value is not None:
                eval_deltas.append((extract_step(record, idx), float(value)))
        elif "n_eval" in record and record.get("n_eval") is not None:
            eval_deltas.append((extract_step(record, idx), float(record["n_eval"])))

    if eval_deltas:
        totals: dict[int, float] = {}
        for step, delta in eval_deltas:
            totals[step] = totals.get(step, 0.0) + delta
        cumulative = []
        running = 0.0
        for step in sorted(totals):
            running += totals[step]
            cumulative.append({"step": step, "score": running})
        return cumulative

    # PT-BR: fallback quando n_eval não está presente, usamos pop_size por geração.
    pop_size: float | None = None
    for record in records:
        if record.get("name") == "pop_size" and record.get("value") is not None:
            pop_size = float(record["value"])
            break
        if record.get("pop_size") is not None:
            pop_size = float(record["pop_size"])
            break
    if pop_size is None:
        pop_size = 1.0

    steps = sorted({extract_step(record, idx) for idx, record in enumerate(records)})
    cumulative = []
    running = 0.0
    for step in steps:
        running += pop_size
        cumulative.append({"step": step, "score": running})
    return cumulative


def extract_best_scores(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Extrai os melhores scores cheap e expensive disponíveis nas métricas."""

    cheap_series, _ = metrics_series(
        records,
        ["best_score", "best_score_cheap", "best"],
        aggregate="min",
    )
    expensive_series, _ = metrics_series(
        records,
        ["best_score_expensive"],
        aggregate="min",
    )
    best_cheap = best_so_far(cheap_series, mode="max")
    best_expensive = best_so_far(expensive_series, mode="max")
    return {
        "best_cheap": best_cheap[-1]["score"] if best_cheap else None,
        "best_expensive": best_expensive[-1]["score"] if best_expensive else None,
    }


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


def _apply_cumulative_best(
    series: list[dict[str, Any]],
    mode: CumulativeMode,
) -> list[dict[str, Any]]:
    """Aplica o melhor-so-far cumulativo, garantindo monotonicidade."""

    best_value: float | None = None
    cumulative: list[dict[str, Any]] = []
    for item in series:
        score = item.get("score")
        if score is None:
            continue
        score_val = float(score)
        if best_value is None:
            best_value = score_val
        elif mode == "min" and score_val < best_value:
            best_value = score_val
        elif mode == "max" and score_val > best_value:
            best_value = score_val
        # PT-BR: garantimos monotonicidade conforme a direção escolhida.
        cumulative.append({"step": int(item["step"]), "score": best_value})
    return cumulative


def _resolve_cumulative_mode(
    selected_key: str | None,
    cumulative_best: bool | CumulativeMode,
) -> CumulativeMode | None:
    if not cumulative_best:
        return None
    if isinstance(cumulative_best, str):
        return cumulative_best
    if selected_key and selected_key in CUMULATIVE_BEST_BY_KEY:
        return CUMULATIVE_BEST_BY_KEY[selected_key]
    return "min"


def metrics_series(
    records: list[dict[str, Any]],
    keys: list[str],
    step_keys: list[str] | None = None,
    aggregate: Literal["min"] | None = None,
    cumulative_best: bool | CumulativeMode = False,
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
        cumulative_mode = _resolve_cumulative_mode(selected_key, cumulative_best)
        if cumulative_mode:
            series = _apply_cumulative_best(series, cumulative_mode)
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
    cumulative_mode = _resolve_cumulative_mode(selected_key, cumulative_best)
    if cumulative_mode:
        series = _apply_cumulative_best(series, cumulative_mode)
    return series, selected_key


def _extract_compare_block(report_data: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Busca o bloco de comparação correspondente ao rótulo fornecido."""

    # PT-BR: tentamos diferentes formatos de relatório aceitos historicamente.
    for container in (report_data, report_data.get("runs"), report_data.get("comparison")):
        if isinstance(container, dict) and label in container and isinstance(container[label], dict):
            return container[label]
    return None


def _load_summary(block: dict[str, Any]) -> dict[str, Any] | None:
    """Carrega o summary.json a partir de summary_path quando disponível."""

    summary_path = block.get("summary_path")
    if not summary_path:
        return None
    try:
        return load_json(Path(summary_path))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _lookup_metric(block: dict[str, Any], keys: Iterable[str], summary: dict[str, Any] | None = None) -> Any:
    """Procura um valor de métrica em um bloco, considerando chaves alternativas."""

    for key in keys:
        if key in block and block[key] is not None:
            return block[key]
    metrics = block.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]
    timing = block.get("timing")
    if isinstance(timing, dict):
        for key in keys:
            if key in timing and timing[key] is not None:
                return timing[key]
    if summary:
        for key in keys:
            if key in summary and summary[key] is not None:
                return summary[key]
    return None


def build_compare_table(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Monta a tabela de comparação entre modos completo e reduzido."""

    rows: list[dict[str, Any]] = []
    for label in ("full", "reduced"):
        block = _extract_compare_block(report_data, label)
        if not block:
            continue
        summary = _load_summary(block)
        label_name = "Completo" if label == "full" else "Reduzido"
        pockets_total = _lookup_metric(
            block,
            ["n_pockets_total", "pockets_total", "n_pockets_detected"],
            summary,
        )
        pockets_used = _lookup_metric(
            block,
            ["n_pockets_used", "pockets_used"],
            summary,
        )
        reduction_ratio = _lookup_metric(block, ["reduction_ratio", "ratio"], summary)
        if reduction_ratio is None and pockets_total:
            try:
                total_val = float(pockets_total)
                used_val = float(pockets_used) if pockets_used is not None else None
                if used_val is not None and total_val:
                    reduction_ratio = used_val / total_val
            except (TypeError, ValueError):
                reduction_ratio = None
        # PT-BR: usamos chaves alternativas para compatibilidade retroativa.
        rows.append(
            {
                "Modo": label_name,
                "Melhor score (cheap)": _lookup_metric(
                    block,
                    ["best_score_cheap", "best_score", "best"],
                    summary,
                ),
                "Avaliações": _lookup_metric(
                    block,
                    ["n_eval", "n_eval_total", "evals", "evaluations"],
                    summary,
                ),
                "Bolsões totais": pockets_total,
                "Bolsões usados": pockets_used,
                "Razão de redução": reduction_ratio,
                "Tempo (s)": _lookup_metric(
                    block,
                    ["elapsed_s", "elapsed_seconds", "elapsed", "total_s"],
                    summary,
                ),
            }
        )
    return rows
