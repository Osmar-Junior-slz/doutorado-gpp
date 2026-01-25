"""Pocket prior model stub."""

from __future__ import annotations

import numpy as np


class PriorNetPocket:
    """Stub prior network for pocket scoring."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return pocket prior scores."""

        return np.zeros(features.shape[0], dtype=float)
