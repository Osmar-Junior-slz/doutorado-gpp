"""Helpers de I/O para o dockingpp."""

# PT-BR: este módulo lida com carregamento de dados e integra o pipeline de
# escaneamento/detecção quando bolsões são solicitados.

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from dockingpp.core.deteccao_bolsoes import construir_bolso_global, detectar_bolsoes
from dockingpp.core.escaneamento_receptor import escanear_receptor
from dockingpp.data.structs import Pocket


def load_config(path: str) -> Dict[str, Any]:
    """Carrega um arquivo de configuração YAML."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_receptor(path: str) -> Any:
    """Carrega dados do receptor a partir do disco."""

    if path == "__dummy__":
        return {"dummy": True}
    return load_pdb_coords(path)


def load_peptide(path: str) -> Any:
    """Carrega dados do peptídeo a partir do disco."""

    if path == "__dummy__":
        return {"dummy": True}
    return load_pdb_coords(path)


def load_pdb_coords(path: str) -> np.ndarray:
    """Carrega coordenadas atômicas de um arquivo PDB.

    PT-BR: lê registros ATOM e HETATM usando colunas padrão do PDB.
    """

    coords: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])

    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.array(coords, dtype=float)


def _extract_coords(receptor: Any) -> np.ndarray:
    """Extrai coordenadas do receptor (PT-BR)."""
    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = getattr(receptor, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _get_cfg_value(cfg: Optional[Any], key: str, default: Any) -> Any:
    """Busca um valor no cfg com fallback (PT-BR)."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_pockets(
    receptor: Any,
    cfg: Optional[Any] = None,
    pockets_path: Optional[str] = None,
) -> List[Pocket]:
    """Inferir bolsões de ligação a partir do receptor (pipeline PhD)."""

    receptor_coords = _extract_coords(receptor)
    scan = escanear_receptor(receptor, cfg=cfg)
    pockets = detectar_bolsoes(scan, cfg=cfg)

    # PT-BR: se houver um arquivo de bolsões, usamos como filtro/seleção
    # baseada nos bolsões detectados (sem criar bolsões padrão).
    if pockets_path and os.path.exists(pockets_path):
        with open(pockets_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        pockets_payload = payload.get("pockets", [])
        if pockets_payload and pockets:
            selected: list[Pocket] = []
            for entry in pockets_payload:
                center = np.asarray(entry.get("center", [0.0, 0.0, 0.0]), dtype=float)
                deltas = np.array([np.linalg.norm(pocket.center - center) for pocket in pockets])
                if deltas.size == 0:
                    continue
                nearest_idx = int(np.argmin(deltas))
                selected.append(pockets[nearest_idx])
            if selected:
                return selected

    if pockets:
        return pockets

    # PT-BR: fallback global explícito somente quando não há bolsões.
    return [construir_bolso_global(receptor_coords, cfg=cfg)]
