"""KD-tree wrappers with a fallback implementation."""

from __future__ import annotations

from typing import Any, List, Optional, Type

import importlib
import importlib.util

import numpy as np


def _get_ckdtree() -> Optional[Type[Any]]:
    spec = importlib.util.find_spec("scipy.spatial")
    if spec is None:
        return None
    module = importlib.import_module("scipy.spatial")
    return getattr(module, "cKDTree", None)


def build_kdtree(points: np.ndarray) -> Any:
    """Build a KD-tree or return the raw points for fallback."""

    ckd = _get_ckdtree()
    if ckd is None:
        return {"points": np.asarray(points)}
    return ckd(points)


def query_radius(tree: Any, points: np.ndarray, r: float) -> List[np.ndarray]:
    """Query neighbors within radius r for each point."""

    pts = np.asarray(points)
    ckd = _get_ckdtree()
    if ckd is not None and isinstance(tree, ckd):
        return [np.asarray(idxs, dtype=int) for idxs in tree.query_ball_point(pts, r)]

    ref = tree["points"]
    results: List[np.ndarray] = []
    for point in pts:
        dists = np.sqrt(np.sum((ref - point) ** 2, axis=-1))
        results.append(np.where(dists <= r)[0])
    return results
