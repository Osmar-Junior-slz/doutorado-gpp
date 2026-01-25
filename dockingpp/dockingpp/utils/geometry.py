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


def apply_transform(
    coords: np.ndarray, rot_matrix: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply a rigid transform to coordinates."""

    rotated = np.asarray(coords, dtype=float) @ np.asarray(rot_matrix, dtype=float).T
    return rotated + np.asarray(translation, dtype=float)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a random rotation matrix."""

    quat = rng.normal(size=4)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3, dtype=float)
    quat = quat / norm
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )
