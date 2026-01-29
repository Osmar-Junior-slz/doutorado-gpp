import numpy as np

from dockingpp.data.io import load_pockets


def test_load_pockets_returns_global_when_no_grid_pockets():
    receptor = {
        "coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=float,
        )
    }
    cfg = {"min_pocket_atoms": 10, "pocket_grid_size": 1.0, "auto_pocket_count": 4}

    pockets = load_pockets(receptor, cfg=cfg)

    assert len(pockets) == 1
    assert pockets[0].id == "global"
    assert len(pockets) != 4
