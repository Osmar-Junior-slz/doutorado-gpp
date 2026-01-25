"""Core data structures for dockingpp."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Pocket:
    """Binding pocket representation."""

    center: np.ndarray
    radius: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose:
    """Pose representation for a peptide."""

    coords: np.ndarray
    score_cheap: Optional[float] = None
    score_expensive: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    """Collection of poses in a generation."""

    poses: List[Pose]
    generation: int


@dataclass
class RunResult:
    """Outcome of a docking run."""

    best_pose: Pose
    population: Optional[Population]
    metrics: Dict[str, Any] = field(default_factory=dict)
