"""Pose prior model stub."""

from __future__ import annotations

import numpy as np


class PriorNetPose:
    """Stub prior network for pose scoring."""

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Return pose prior scores."""

        return np.zeros(coords.shape[0], dtype=float)
