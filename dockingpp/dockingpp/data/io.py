"""I/O helpers for dockingpp."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from dockingpp.data.structs import Pocket


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_receptor(path: str) -> Any:
    """Load receptor data from disk."""

    if path == "__dummy__":
        return {"dummy": True}
    return load_pdb_coords(path)


def load_peptide(path: str) -> Any:
    """Load peptide data from disk."""

    if path == "__dummy__":
        return {"dummy": True}
    return load_pdb_coords(path)


def load_pdb_coords(path: str) -> np.ndarray:
    """Load atomic coordinates from a PDB file.

    Reads ATOM and HETATM records using standard PDB column positions.
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
    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = getattr(receptor, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _get_cfg_value(cfg: Optional[Any], key: str, default: Any) -> Any:
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
    """Infer binding pockets from a receptor."""

    receptor_coords = _extract_coords(receptor)
    pocket_margin = float(_get_cfg_value(cfg, "pocket_margin", 2.0))
    if pockets_path and os.path.exists(pockets_path):
        with open(pockets_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        pockets_payload = payload.get("pockets", [])
        pockets: list[Pocket] = []
        for idx, entry in enumerate(pockets_payload):
            center = np.asarray(entry.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(entry.get("radius", 0.0))
            pocket_id = str(entry.get("id") or f"pocket_{idx}")
            if receptor_coords.size:
                deltas = receptor_coords - center.reshape(1, 3)
                dists = np.linalg.norm(deltas, axis=1)
                mask = dists <= radius
                pocket_coords = receptor_coords[mask]
            else:
                pocket_coords = receptor_coords.copy()
            pockets.append(
                Pocket(
                    id=pocket_id,
                    center=center,
                    radius=radius,
                    coords=pocket_coords,
                    meta={"coords": pocket_coords},
                )
            )
        return pockets

    if receptor_coords.size:
        center = receptor_coords.mean(axis=0)
        deltas = receptor_coords - center.reshape(1, 3)
        max_dist = float(np.max(np.linalg.norm(deltas, axis=1)))
    else:
        center = np.zeros(3, dtype=float)
        max_dist = 0.0
    radius = max_dist + pocket_margin
    pocket_coords = receptor_coords.copy()
    return [
        Pocket(
            id="global",
            center=center,
            radius=radius,
            coords=pocket_coords,
            meta={"coords": pocket_coords},
        )
    ]
