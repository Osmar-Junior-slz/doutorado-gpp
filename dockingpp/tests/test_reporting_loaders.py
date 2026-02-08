"""Testes para loaders de relatórios (PT-BR)."""

from __future__ import annotations

import json
from pathlib import Path

from dockingpp.reporting.loaders import extract_series, load_jsonl


def test_load_jsonl_tolerante_e_extract_series_com_faltas(tmp_path: Path) -> None:
    """Garante que JSONL inválido não quebra e que faltas são registradas."""

    jsonl_path = tmp_path / "metrics.jsonl"
    linhas = [
        json.dumps({"step": 0, "best_score_cheap": 1.0}),
        "linha invalida",
        json.dumps({"generation": 1, "n_eval_total": 5}),
    ]
    jsonl_path.write_text("\n".join(linhas), encoding="utf-8")

    eventos = load_jsonl(jsonl_path)
    assert len(eventos) == 2

    series = extract_series(eventos)
    assert "missing" in series
    assert series["iter"] == [0, 1]
    assert len(series["best_cheap"]) == 2
    assert series["missing"]["best_expensive"] == 2
