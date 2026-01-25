"""I/O helpers for dockingpp."""

from __future__ import annotations

from typing import Any, Dict, List

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


def load_pockets(receptor: Any) -> List[Pocket]:
    """Infer binding pockets from a receptor."""

    raise NotImplementedError("Pocket detection is not implemented yet.")
