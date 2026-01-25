"""Confidence feature extraction stubs."""

from __future__ import annotations

from typing import Dict

from dockingpp.data.structs import Pose


def extract_confidence_features(pose: Pose) -> Dict[str, float]:
    """Extract confidence-related features for a pose."""

    return {"score_cheap": pose.score_cheap or 0.0}
