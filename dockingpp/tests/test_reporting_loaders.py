"""Testes para loaders de relatórios (PT-BR)."""

from __future__ import annotations

import json
from pathlib import Path

from dockingpp.reporting.loaders import (
    extract_series,
    find_report_runs,
    load_jsonl,
    pair_full_reduced,
)


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


def test_find_report_runs_full_reduced_pair(tmp_path: Path) -> None:
    """Garante que runs full/reduced são identificados e pareados."""

    full_dir = tmp_path / "full_run"
    reduced_dir = tmp_path / "reduced_run"
    full_dir.mkdir()
    reduced_dir.mkdir()

    (full_dir / "config_any.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "generations": 10,
                "pop_size": 20,
                "full_search": True,
                "search_space_mode": "global",
            }
        ),
        encoding="utf-8",
    )
    (full_dir / "summary_any.json").write_text(
        json.dumps(
            {
                "timing": {"total_s": 12.0},
                "best_score_cheap": 0.9,
                "mode": "single",
                "search_space_mode": "global",
            }
        ),
        encoding="utf-8",
    )
    (full_dir / "result_any.json").write_text(
        json.dumps(
            {
                "run_id": "full-1",
                "best_cheap_by_pocket": {},
                "n_eval_total": 100,
            }
        ),
        encoding="utf-8",
    )
    (full_dir / "metrics.jsonl").write_text(json.dumps({"step": 0, "best_score_cheap": 0.9}), encoding="utf-8")

    (reduced_dir / "config_other.json").write_text(
        json.dumps(
            {
                "seed": 2,
                "generations": 5,
                "pop_size": 10,
                "full_search": False,
                "search_space_mode": "pockets",
            }
        ),
        encoding="utf-8",
    )
    (reduced_dir / "summary_other.json").write_text(
        json.dumps(
            {
                "timing": {"total_s": 4.0},
                "best_score_cheap": 0.7,
                "mode": "single",
                "search_space_mode": "pockets",
            }
        ),
        encoding="utf-8",
    )
    (reduced_dir / "result_other.json").write_text(
        json.dumps(
            {
                "run_id": "reduced-1",
                "best_cheap_by_pocket": {},
                "n_eval_total": 50,
            }
        ),
        encoding="utf-8",
    )

    runs = find_report_runs(tmp_path)
    assert len(runs) == 2
    kinds = {run.kind for run in runs}
    assert kinds == {"full", "reduced"}

    full_run, reduced_run = pair_full_reduced(runs)
    assert full_run is not None
    assert reduced_run is not None
    assert full_run.kind == "full"
    assert reduced_run.kind == "reduced"
