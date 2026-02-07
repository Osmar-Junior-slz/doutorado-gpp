import numpy as np

from dockingpp.data.structs import Pose
from dockingpp.scoring.expensive import score_pose_expensive


def test_expensive_score_basic_monotonicity():
    receptor = {"coords": np.array([[0.0, 0.0, 0.0]], dtype=float)}
    cfg = {
        "expensive": {
            "contact_cutoff": 4.0,
            "clash_cutoff": 2.0,
            "w_att": 1.0,
            "w_rep": 3.0,
        }
    }
    pose_near = Pose(coords=np.array([[3.5, 0.0, 0.0]], dtype=float))
    pose_clash = Pose(coords=np.array([[1.0, 0.0, 0.0]], dtype=float))

    score_near = score_pose_expensive(pose_near, receptor, None, cfg)
    score_clash = score_pose_expensive(pose_clash, receptor, None, cfg)

    assert score_near is not None
    assert score_clash is not None
    assert score_clash > score_near
