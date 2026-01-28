"""Responsável por escanear o receptor e extrair evidências geométrico-químicas."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _extrair_coords_receptor(receptor: Any) -> np.ndarray:
    """Extrai coordenadas atômicas do receptor em formato NumPy.

    PT-BR: suportamos dict, np.ndarray ou objetos com atributo "coords".
    """

    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = getattr(receptor, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def escanear_receptor(receptor: Any, cfg: Any | None = None) -> Dict[str, Any]:
    """Escaneia o receptor e retorna evidências geométricas básicas.

    PT-BR: o scan é a primeira etapa do pipeline. Fornece coordenadas, caixa
    delimitadora e centro geométrico para etapas de detecção de bolsões.
    """

    _ = cfg
    coords = _extrair_coords_receptor(receptor)
    if coords.size == 0:
        return {
            "coords": coords,
            "bbox_min": None,
            "bbox_max": None,
            "center": None,
            "scan_ok": False,
        }

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    center = coords.mean(axis=0)

    return {
        "coords": coords,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "center": center,
        "scan_ok": True,
    }


def scan_receptor(receptor: Any, cfg: Any | None = None) -> Dict[str, Any]:
    """Alias retrocompatível para escanear o receptor.

    PT-BR: mantém compatibilidade para futuras integrações em inglês.
    """

    return escanear_receptor(receptor, cfg=cfg)
