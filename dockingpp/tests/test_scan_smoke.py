import numpy as np

from dockingpp.data.structs import Pocket
from dockingpp.pipeline.scan import (
    build_receptor_kdtree,
    scan_pocket_feasibility,
    select_pockets_from_scan,
)


def test_scan_smoke_selects_less_clashy_pocket():
    receptor_coords = np.array(
        [
            [x, y, z]
            for x in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for z in (-1.0, 0.0, 1.0)
        ],
        dtype=float,
    )
    peptide_coords = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [-0.2, 0.0, 0.0],
        ],
        dtype=float,
    )

    pocket_close = Pocket(
        id="close",
        center=np.array([0.0, 0.0, 0.0], dtype=float),
        radius=3.0,
        coords=receptor_coords,
    )
    pocket_far = Pocket(
        id="far",
        center=np.array([15.0, 15.0, 15.0], dtype=float),
        radius=3.0,
        coords=receptor_coords,
    )

    scan_cfg = {
        "samples_per_pocket": 32,
        "clash_cutoff": 1.5,
        "contact_cutoff": 4.0,
        "max_clash_ratio": 0.01,
    }
    rng = np.random.default_rng(2024)
    receptor_tree = build_receptor_kdtree(receptor_coords)

    metrics_close = scan_pocket_feasibility(
        receptor_tree,
        peptide_coords,
        pocket_close,
        scan_cfg,
        rng,
    )
    metrics_far = scan_pocket_feasibility(
        receptor_tree,
        peptide_coords,
        pocket_far,
        scan_cfg,
        rng,
    )

    assert metrics_far["feasible_fraction"] > metrics_close["feasible_fraction"]
    assert metrics_far["scan_score"] >= metrics_close["scan_score"]

    scan_table = {"close": metrics_close, "far": metrics_far}
    selected = select_pockets_from_scan([pocket_close, pocket_far], scan_table, top_k=1)

    assert len(selected) == 1
    assert selected[0].id == "far"
