"""Testes de regressão para séries de métricas e progresso (PT-BR)."""

from __future__ import annotations

import json
from pathlib import Path

from dockingpp.gui.services.progress_service import (
    compute_progress,
    format_progress_text,
    read_last_metrics_generation,
)
from dockingpp.gui.services.report_service import metrics_series


def test_series_remove_serrilhado_com_steps_duplicados() -> None:
    """Garante que a série agregada elimine steps duplicados (serrilhado)."""

    # PT-BR: dois registros com o mesmo step simulam duplicação por pocket.
    records = [
        {"name": "best_score", "value": 10.0, "step": 0},
        {"name": "best_score", "value": 8.0, "step": 0},
        {"name": "best_score", "value": 9.0, "step": 1},
    ]

    series, _ = metrics_series(records, ["best_score"], aggregate="min", cumulative_best=True)
    steps = [item["step"] for item in series]

    # PT-BR: a série deve conter apenas steps únicos após agregação.
    assert len(steps) == len(set(steps))


def test_convergencia_best_so_far_eh_monotona_nao_crescente() -> None:
    """Valida que o melhor-so-far é monotônico (não crescente)."""

    records = [
        {"name": "best_score", "value": 5.0, "step": 0},
        {"name": "best_score", "value": 7.0, "step": 1},
        {"name": "best_score", "value": 3.0, "step": 2},
    ]

    series, _ = metrics_series(records, ["best_score"], aggregate="min", cumulative_best=True)
    scores = [item["score"] for item in series]

    # PT-BR: o best-so-far deve apenas diminuir ou permanecer igual.
    assert all(scores[idx] >= scores[idx + 1] for idx in range(len(scores) - 1))


def test_progresso_nao_extrapola_total_de_geracoes(tmp_path: Path) -> None:
    """Assegura que geração e progresso respeitam o limite de gerações."""

    # PT-BR: simulamos metrics.jsonl com step global e total_generations conhecido.
    metrics_path = tmp_path / "metrics.jsonl"
    payload = {"name": "best_score", "value": 1.0, "step": 12, "total_generations": 5}
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    generation = read_last_metrics_generation(metrics_path)
    progress = compute_progress(generation, total_generations=5)
    progress_text = format_progress_text(generation, total_generations=5, progress=progress)

    # PT-BR: a geração derivada deve estar entre 0 e total_generations.
    assert generation is not None
    assert 0 <= generation <= 5
    # PT-BR: o texto não pode exibir um contador acima do total.
    assert f"Geração {generation} / 5" in progress_text

    # PT-BR: o clamp evita que um valor acima do total apareça no texto.
    progress_clamped = compute_progress(12, total_generations=5)
    progress_text_clamped = format_progress_text(12, total_generations=5, progress=progress_clamped)
    assert progress_clamped == 1.0
    assert "Geração 5 / 5" in progress_text_clamped
