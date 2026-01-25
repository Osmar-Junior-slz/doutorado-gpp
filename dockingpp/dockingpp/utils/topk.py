"""Utilities for top-k selection."""

from __future__ import annotations

import numpy as np


def topk_indices(values: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Return indices for the top-k values in sorted order."""

    if k <= 0:
        return np.array([], dtype=int)

    n = values.shape[0]
    k = min(k, n)
    if largest:
        idx = np.argpartition(-values, k - 1)[:k]
        order = np.argsort(-values[idx])
    else:
        idx = np.argpartition(values, k - 1)[:k]
        order = np.argsort(values[idx])
    return idx[order]
