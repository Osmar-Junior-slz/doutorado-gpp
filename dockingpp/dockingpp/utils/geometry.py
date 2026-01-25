"""Geometry helpers."""

from __future__ import annotations

import numpy as np


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""

    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""

    diff = a - b
    return float(np.sqrt(np.mean(diff**2)))
