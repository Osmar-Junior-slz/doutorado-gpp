"""Confidence calibration stub."""

from __future__ import annotations

from typing import Any, Dict


class ConfidenceCalibrator:
    """Placeholder confidence calibrator."""

    def calibrate(self, features: Dict[str, float]) -> float:
        """Return a dummy confidence score."""

        _ = features
        return 0.0

    def fit(self, data: Any) -> None:
        """Fit the calibrator (stub)."""

        _ = data
