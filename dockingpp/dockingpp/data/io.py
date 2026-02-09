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


def _get_debug_logger(cfg: Optional[Any]) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get("debug_logger")
    return getattr(cfg, "debug_logger", None)


def load_pockets(
    receptor: Any,
    cfg: Optional[Any] = None,
    pockets_path: Optional[str] = None,
) -> List[Pocket]:
    """Infer binding pockets from a receptor."""

    receptor_coords = _extract_coords(receptor)
    debug_logger = _get_debug_logger(cfg)
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
        if debug_logger is not None:
            debug_logger.log(
                {
                    "type": "pocket_generation",
                    "mode": "file",
                    "pockets_path": pockets_path,
                    "receptor_coords_n": int(receptor_coords.shape[0]),
                    "n_pockets": int(len(pockets)),
                }
            )
        return pockets

    def build_pocket(pocket_id: str, coords: np.ndarray) -> Pocket:
        if coords.size:
            center = coords.mean(axis=0)
            deltas = coords - center.reshape(1, 3)
            max_dist = float(np.max(np.linalg.norm(deltas, axis=1)))
        else:
            center = np.zeros(3, dtype=float)
            max_dist = 0.0
        radius = max_dist + pocket_margin
        return Pocket(
            id=pocket_id,
            center=center,
            radius=radius,
            coords=coords,
            meta={"coords": coords},
        )

    if receptor_coords.size:
        default_min_atoms = max(5, int(receptor_coords.shape[0] * 0.02))
    else:
        default_min_atoms = 0
    min_pocket_atoms = int(_get_cfg_value(cfg, "min_pocket_atoms", default_min_atoms) or 0)
    grid_size = float(_get_cfg_value(cfg, "pocket_grid_size", 8.0))
    pockets: list[Pocket] = []
    bbox_min = np.zeros(3, dtype=float)
    bbox_max = np.zeros(3, dtype=float)
    bbox_range = np.zeros(3, dtype=float)
    grid_padding = float(_get_cfg_value(cfg, "pocket_grid_padding", 0.0) or 0.0)
    nx = ny = nz = 0
    n_grid_points_total = 0
    n_candidates_before_filter = 0
    n_after_filter_min_atoms: int | None = None
    n_after_filter_density: int | None = None
    n_after_clustering: int | None = None

    def _log_grid_stats(n_pockets_final: int) -> None:
        if debug_logger is None:
            return
        debug_logger.log(
            {
                "type": "pocket_grid_stats",
                "out_dir": _get_cfg_value(cfg, "out_dir", None),
                "mode": "grid",
                "receptor_coords_n": int(receptor_coords.shape[0]),
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "bbox_range": bbox_range.tolist(),
                "padding": float(grid_padding),
                "step": float(grid_size),
                "nx": int(nx),
                "ny": int(ny),
                "nz": int(nz),
                "n_grid_points_total": int(n_grid_points_total),
                "n_candidates_before_filter": int(n_candidates_before_filter),
                "n_after_filter_density": (
                    int(n_after_filter_density) if n_after_filter_density is not None else None
                ),
                "n_after_filter_min_atoms": (
                    int(n_after_filter_min_atoms) if n_after_filter_min_atoms is not None else None
                ),
                "n_after_clustering": (
                    int(n_after_clustering) if n_after_clustering is not None else None
                ),
                "n_pockets_final": int(n_pockets_final),
            }
        )

    def _log_grid_failure(reason: str, n_pockets_final: int, recovered: bool = False) -> None:
        if debug_logger is None:
            return
        debug_logger.log(
            {
                "type": "pocket_grid_failure",
                "reason": reason,
                "recovered": bool(recovered),
                "mode": "grid",
                "receptor_coords_n": int(receptor_coords.shape[0]),
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "bbox_range": bbox_range.tolist(),
                "padding": float(grid_padding),
                "step": float(grid_size),
                "nx": int(nx),
                "ny": int(ny),
                "nz": int(nz),
                "n_grid_points_total": int(n_grid_points_total),
                "n_candidates_before_filter": int(n_candidates_before_filter),
                "n_after_filter_density": (
                    int(n_after_filter_density) if n_after_filter_density is not None else None
                ),
                "n_after_filter_min_atoms": (
                    int(n_after_filter_min_atoms) if n_after_filter_min_atoms is not None else None
                ),
                "n_after_clustering": (
                    int(n_after_clustering) if n_after_clustering is not None else None
                ),
                "n_pockets_final": int(n_pockets_final),
            }
        )

    grid_failed = False

    if receptor_coords.size and grid_size <= 0:
        _log_grid_failure("step_invalid", 0)
        grid_failed = True

    if receptor_coords.size and grid_size > 0:
        bbox_min = receptor_coords.min(axis=0) - grid_padding
        bbox_max = receptor_coords.max(axis=0) + grid_padding
        bbox_range = bbox_max - bbox_min
        if np.any(bbox_range <= 0.0):
            _log_grid_failure("bbox_invalid", 0)
            grid_failed = True
        if not grid_failed:
            nx = max(1, int(np.floor(bbox_range[0] / grid_size)) + 1)
            ny = max(1, int(np.floor(bbox_range[1] / grid_size)) + 1)
            nz = max(1, int(np.floor(bbox_range[2] / grid_size)) + 1)
            n_grid_points_total = int(nx * ny * nz)
            if n_grid_points_total <= 0:
                _log_grid_failure("grid_empty", 0)
                grid_failed = True
            if not grid_failed:
                min_coord = bbox_min
                indices = np.floor((receptor_coords - min_coord.reshape(1, 3)) / grid_size).astype(
                    int
                )
                groups: dict[tuple[int, int, int], list[int]] = {}
                for idx, cell in enumerate(indices):
                    key = (int(cell[0]), int(cell[1]), int(cell[2]))
                    groups.setdefault(key, []).append(idx)
                n_candidates_before_filter = len(groups)
                for pocket_idx, (cell, atom_indices) in enumerate(sorted(groups.items())):
                    if min_pocket_atoms and len(atom_indices) < min_pocket_atoms:
                        continue
                    coords = receptor_coords[atom_indices]
                    pockets.append(build_pocket(f"auto_grid_{pocket_idx}", coords))
                n_after_filter_min_atoms = len(pockets)
                n_after_filter_density = n_after_filter_min_atoms
                n_after_clustering = n_after_filter_min_atoms
                if not pockets and n_candidates_before_filter > 0 and min_pocket_atoms:
                    largest_cell = max(groups.items(), key=lambda item: len(item[1]))
                    coords = receptor_coords[largest_cell[1]]
                    pockets.append(build_pocket("auto_grid_0", coords))
                    n_after_filter_min_atoms = len(pockets)
                    n_after_filter_density = n_after_filter_min_atoms
                    n_after_clustering = n_after_filter_min_atoms
                    _log_grid_failure("filters_removed_all", len(pockets), recovered=True)
                    grid_failed = False

    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "pocket_generation",
                "mode": "grid",
                "pockets_path": pockets_path,
                "receptor_coords_n": int(receptor_coords.shape[0]),
                "n_pockets": int(len(pockets)),
            }
        )
        _log_grid_stats(len(pockets))

    if len(pockets) >= 1:
        return pockets

    if debug_logger is not None:
        reason = "no_coords" if receptor_coords.size == 0 else "<=1_pocket"
        debug_logger.log(
            {"type": "pocket_fallback", "reason": reason, "n_generated": int(len(pockets))}
        )
        if receptor_coords.size and not grid_failed:
            _log_grid_failure("unknown", 0)
    pocket_coords = receptor_coords.copy()
    return [build_pocket("global", pocket_coords)]
