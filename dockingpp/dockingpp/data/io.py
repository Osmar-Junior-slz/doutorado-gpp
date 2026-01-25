"""I/O helpers for dockingpp."""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from dockingpp.data.structs import Pocket


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_receptor(path: str) -> Any:
    """Load receptor data from disk."""

    raise NotImplementedError("Receptor parsing is not implemented yet.")


def load_peptide(path: str) -> Any:
    """Load peptide data from disk."""

    raise NotImplementedError("Peptide parsing is not implemented yet.")


def load_pockets(receptor: Any) -> List[Pocket]:
    """Infer binding pockets from a receptor."""

    raise NotImplementedError("Pocket detection is not implemented yet.")
