import numpy as np

from dockingpp.utils.topk import topk_indices


def test_topk_largest():
    values = np.array([1.0, 3.0, 2.0, 5.0])
    idx = topk_indices(values, k=2, largest=True)
    assert idx.tolist() == [3, 1]


def test_topk_smallest():
    values = np.array([1.0, 3.0, 2.0, 5.0])
    idx = topk_indices(values, k=3, largest=False)
    assert idx.tolist() == [0, 2, 1]


def test_topk_all():
    values = np.array([4.0, 2.0, 7.0])
    idx = topk_indices(values, k=3, largest=True)
    assert idx.tolist() == [2, 0, 1]
