import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.priors.pocket import rank_pockets


def _make_coords(center, spread, count=100):
    axis = np.linspace(-spread, spread, count)
    coords = np.column_stack(
        (
            center[0] + axis,
            center[1] + axis[::-1],
            center[2] + np.sin(axis),
        )
    )
    return coords.astype(float)


def test_rank_pockets_prefers_larger_and_more_compact():
    center = np.zeros(3, dtype=float)
    compact_coords = _make_coords(center, spread=0.5)
    spread_coords = _make_coords(center, spread=5.0)

    pocket_compact = Pocket(
        id="compact",
        center=center,
        radius=6.0,
        coords=compact_coords,
    )
    pocket_spread = Pocket(
        id="spread",
        center=center,
        radius=6.0,
        coords=spread_coords,
    )

    ranked = rank_pockets({}, [pocket_spread, pocket_compact])

    assert ranked[0][0].id == "compact"


def test_rank_pockets_proximity_if_peptide_given():
    near_center = np.array([0.0, 0.0, 0.0])
    far_center = np.array([10.0, 0.0, 0.0])
    pocket_near = Pocket(
        id="near",
        center=near_center,
        radius=6.0,
        coords=_make_coords(near_center, spread=1.0),
    )
    pocket_far = Pocket(
        id="far",
        center=far_center,
        radius=6.0,
        coords=_make_coords(far_center, spread=1.0),
    )

    receptor = {
        "coords": np.zeros((0, 3), dtype=float),
        "pocket_rank_weights": {
            "w_size": 0.0,
            "w_compact": 0.0,
            "w_depth": 0.0,
            "w_proximity": 1.0,
        },
    }
    peptide = {"coords": np.array([[0.5, 0.0, 0.0]], dtype=float)}

    ranked = rank_pockets(receptor, [pocket_far, pocket_near], peptide=peptide)

    assert ranked[0][0].id == "near"
