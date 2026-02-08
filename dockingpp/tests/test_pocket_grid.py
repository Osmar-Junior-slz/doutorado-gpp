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
    pockets_repeat = load_pockets(receptor, cfg=cfg)

    assert len(pockets) >= 1
    assert any(pocket.id != "global" for pocket in pockets)
    assert any(pocket.id.startswith("auto_grid_") for pocket in pockets)
    assert [pocket.id for pocket in pockets] == [pocket.id for pocket in pockets_repeat]
    assert [pocket.center.tolist() for pocket in pockets] == [
        pocket.center.tolist() for pocket in pockets_repeat
    ]
