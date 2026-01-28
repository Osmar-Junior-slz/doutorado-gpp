"""Testes da estratégia PhD (scan -> bolsões -> ranking -> docking)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dockingpp.core.deteccao_bolsoes import detectar_bolsoes
from dockingpp.core.escaneamento_receptor import escanear_receptor
from dockingpp.data.structs import Pocket
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.search.abc_ga_vgos import ABCGAVGOSSearch


def _make_receptor(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Cria um receptor simples em formato dict para testes (PT-BR)."""

    return {"coords": coords}


def _make_dummy_logger() -> object:
    """Cria um logger mínimo com log_metric para o motor de busca."""

    class DummyLogger:
        def log_metric(self, name: str, value: float, step: int, extra: dict | None = None) -> None:
            _ = (name, value, step, extra)

    return DummyLogger()


def test_bolsoes_dependem_da_geometria_do_receptor() -> None:
    """Altera coordenadas do receptor e espera mudança nos bolsões."""

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    shift = np.array([10.0, 0.0, 0.0], dtype=float)

    cfg = {
        "pocket_grid_spacing": 2.0,
        "pocket_min_dist": 1.0,
        "pocket_max_dist": 4.0,
        "pocket_min_cluster_points": 1,
    }

    scan1 = escanear_receptor(_make_receptor(coords), cfg=cfg)
    scan2 = escanear_receptor(_make_receptor(coords + shift), cfg=cfg)

    pockets1 = detectar_bolsoes(scan1, cfg=cfg)
    pockets2 = detectar_bolsoes(scan2, cfg=cfg)

    assert pockets1
    assert pockets2

    center1 = np.mean([pocket.center for pocket in pockets1], axis=0)
    center2 = np.mean([pocket.center for pocket in pockets2], axis=0)

    assert np.linalg.norm(center2 - center1) > 5.0


def test_fallback_so_acontece_quando_nao_ha_bolsoes(tmp_path: Path) -> None:
    """Fallback global deve ocorrer apenas quando detecção retorna zero."""

    cfg = Config()
    receptor_path = tmp_path / "empty_receptor.pdb"
    peptide_path = tmp_path / "empty_peptide.pdb"
    receptor_path.write_text("REMARK empty receptor\n", encoding="utf-8")
    peptide_path.write_text("REMARK empty peptide\n", encoding="utf-8")

    out_dir = tmp_path / "out_empty"
    run_pipeline(cfg, str(receptor_path), str(peptide_path), str(out_dir))

    metrics = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    metric_map = {entry["name"]: entry["value"] for entry in metrics}

    assert metric_map.get("pocket_fallback_used") == 1.0
    assert metric_map.get("n_pockets_detected") == 0.0


def test_sem_fallback_quando_bolsoes_detectados(tmp_path: Path) -> None:
    """Não usa bolso global quando há bolsões detectados."""

    cfg = Config()
    out_dir = tmp_path / "out_dummy"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    metrics = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    metric_map = {entry["name"]: entry["value"] for entry in metrics}

    assert metric_map.get("pocket_fallback_used") == 0.0
    assert metric_map.get("n_pockets_detected", 0.0) > 0.0


def test_reduced_seleciona_top_k_menor_que_detectados(tmp_path: Path) -> None:
    """Reduzido deve selecionar no máximo top_k bolsões detectados."""

    cfg = Config(full_search=False, top_pockets=1)
    out_dir = tmp_path / "out_reduced"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    metrics = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    metric_map = {entry["name"]: entry["value"] for entry in metrics}

    detected = int(metric_map.get("n_pockets_detected", 0))
    selected = int(metric_map.get("n_pockets_selected", 0))

    assert selected == min(1, detected) if detected > 0 else selected == 1


def test_busca_usa_todos_os_bolsoes_selecionados() -> None:
    """Garante que o motor de busca percorre todos os bolsões recebidos."""

    cfg = Config(generations=1, pop_size=2)
    search = ABCGAVGOSSearch(cfg)

    receptor = _make_receptor(np.array([[0.0, 0.0, 0.0]], dtype=float))
    peptide = np.array([[0.0, 0.0, 0.0]], dtype=float)

    pockets = [
        Pocket(id="pocket_a", center=np.zeros(3), radius=2.0, coords=np.zeros((1, 3))),
        Pocket(id="pocket_b", center=np.ones(3), radius=2.0, coords=np.zeros((1, 3))),
    ]

    def score_cheap(pose, pocket, weights):
        _ = (pose, weights)
        return 1.0 if pocket.id == "pocket_a" else 10.0

    def score_expensive(pose, receptor_obj, peptide_obj, cfg_obj):
        _ = (pose, receptor_obj, peptide_obj, cfg_obj)
        return 0.0

    result = search.search(
        receptor=receptor,
        peptide=peptide,
        pockets=pockets,
        cfg=cfg,
        score_cheap_fn=score_cheap,
        score_expensive_fn=score_expensive,
        prior_pocket=None,
        prior_pose=None,
        logger=_make_dummy_logger(),
    )

    assert result.best_pose.meta.get("pocket_id") == "pocket_b"
