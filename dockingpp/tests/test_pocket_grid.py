import numpy as np

from dockingpp.data.io import load_pockets


def test_load_pockets_grid_generates_non_global():
    coords = np.array(
        [[x, y, z] for x in range(0, 11, 2) for y in range(0, 11, 2) for z in range(0, 11, 2)],
        dtype=float,
    )
    receptor = {"coords": coords}
    cfg = {"pocket_grid_size": 4.0, "min_pocket_atoms": 1}

    pockets = load_pockets(receptor, cfg=cfg)

    assert len(pockets) >= 1
    assert all(pocket.id != "global" for pocket in pockets)
