"""Serviço de progresso para leitura de métricas e formatação (PT-BR)."""

from __future__ import annotations

import json
from pathlib import Path


def read_last_metrics_generation(metrics_path: Path, max_bytes: int = 8192) -> int | None:
    """Lê o metrics.jsonl incremental para recuperar a geração mais recente.

    PT-BR: o bug "Geração 399 / 100" ocorria porque a UI usava o "step" global
    (que inclui offsets por pocket ou avaliações internas). Agora priorizamos
    o campo "generation" (0..N) gravado a cada geração, garantindo progresso
    correto e sem extrapolar cfg.generations. Caso o "generation" não exista,
    derivamos a geração via (step % total_generations), preservando o significado
    científico quando o step é global. O "step" segue existindo apenas para séries.
    """

    try:
        if not metrics_path.exists():
            return None
        with metrics_path.open("rb") as handle:
            try:
                handle.seek(0, 2)
                file_size = handle.tell()
                read_size = min(file_size, max_bytes)
                handle.seek(-read_size, 2)
            except OSError:
                handle.seek(0)
            chunk = handle.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None

    generations: list[int] = []
    steps: list[int] = []
    for line in chunk.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        generation = record.get("generation")
        total_generations = record.get("total_generations")
        step = record.get("step")
        if isinstance(generation, (int, float)):
            # PT-BR: geração explícita sempre tem prioridade sobre step global.
            generations.append(int(generation))
        elif isinstance(step, (int, float)) and isinstance(total_generations, (int, float)):
            total_int = int(total_generations)
            if total_int > 0:
                # PT-BR: derivamos geração local para evitar extrapolar N.
                generations.append(int(step) % total_int)
        if isinstance(step, (int, float)):
            steps.append(int(step))
    # PT-BR: usamos a última geração encontrada; se não existir, caímos no step.
    if generations:
        return generations[-1]
    return steps[-1] if steps else None


def compute_progress(generation: int | None, total_generations: int) -> float:
    """Calcula a fração de progresso com clamp para evitar extrapolação."""

    if total_generations <= 0:
        return 0.0
    # PT-BR: clamp para evitar valores acima de N mesmo que existam outros contadores.
    current = max(generation or 0, 0)
    current = min(current, total_generations)
    return min(current / total_generations, 1.0)


def format_progress_text(generation: int | None, total_generations: int, progress: float) -> str:
    """Formata o texto de progresso garantindo limites corretos."""

    # PT-BR: usamos a geração normalizada para garantir "Geração g / N" correto.
    current = max(generation or 0, 0)
    current = min(current, total_generations)
    return f"Geração {current} / {total_generations} ({progress * 100:.1f}%)"
