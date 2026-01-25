import numpy as np

from dockingpp.data.structs import Pocket, Pose
from dockingpp.scoring.cheap import score_pose_cheap


def test_score_pose_cheap_counts_contacts_and_clashes():
    pocket = Pocket(
        id="test-pocket",
        center=np.zeros(3, dtype=float),
        radius=6.0,
        coords=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        meta={"coords": np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])},
    )
    pose = Pose(coords=np.array([[0.0, 0.0, 1.0], [4.0, 0.0, 0.0]]))
    weights = {"w_contact": 1.0, "w_clash": 2.0}

    score = score_pose_cheap(pose, pocket, weights)

    assert score == -1.0
