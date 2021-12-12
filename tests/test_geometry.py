import numpy as np

from vascpy.utils import geometry as _geom


def test_sort_points():
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.5, 0.1, 0.5],
            [0.3, 0.3, 0.3],
            [0.5, 0.1, 0.5],
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0],
        ]
    )

    sorted_idx = _geom.sort_points(points)

    expected_idx = [6, 0, 4, 5, 2, 1, 3]

    np.testing.assert_allclose(sorted_idx, expected_idx)


def test_are_consecutive_pairs_different():

    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.3, 0.3],
            [0.1, 0.3, 0.4],
            [0.2, 0.3, 0.4],
            [0.3, 0.2, 0.1],
            [0.3, 0.3, 0.3],
            [0.3, 0.3, 0.3],
            [0.5, 0.1, 0.5],
            [0.5, 0.1, 0.5],
            [0.6, 0.6, 0.5],
        ]
    )

    expected_mask = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1])

    mask = _geom.are_consecutive_pairs_different(points)

    np.testing.assert_allclose(mask, expected_mask)


def test_unique_points__ascending():

    points = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.4], [0.1, 0.2, 0.5]])

    expected_idx = np.array([0, 1, 2])
    expected_mapping = np.array([0, 1, 2])

    idx, mapping = _geom.unique_points(points, decimals=2)

    np.testing.assert_allclose(idx, expected_idx)
    np.testing.assert_allclose(mapping, expected_mapping)


def test_unique_points__descending():

    points = np.array([[0.1, 0.2, 0.5], [0.1, 0.2, 0.4], [0.1, 0.2, 0.3]])

    expected_idx = np.array([0, 1, 2])
    expected_mapping = np.array([0, 1, 2])

    idx, mapping = _geom.unique_points(points, decimals=2)

    np.testing.assert_allclose(idx, expected_idx)
    np.testing.assert_allclose(mapping, expected_mapping)


def test_unique_points__mixed_1():

    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.5, 0.1, 0.5],
            [0.3, 0.3, 0.3],
            [0.5, 0.1, 0.5],
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0],
        ]
    )

    expected_idx = np.array([0, 1, 2, 5, 6])
    expected_mapping = np.array([0, 1, 2, 1, 0, 3, 4])

    idx, mapping = _geom.unique_points(points, decimals=2)

    np.testing.assert_allclose(idx, expected_idx)
    np.testing.assert_allclose(mapping, expected_mapping)


def test_unique_points__mixed_2():

    points = np.array(
        [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.3, 0.4, 0.5]]
    )

    expected_idx = np.array([0, 1, 3])
    expected_mapping = np.array([0, 1, 0, 2, 1])

    idx, mapping = _geom.unique_points(points, decimals=2)

    np.testing.assert_allclose(idx, expected_idx)
    np.testing.assert_allclose(mapping, expected_mapping)


def test_unique_points__an_army_of_clones():

    points = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    expected_idx = np.array([0])
    expected_mapping = np.array([0, 0, 0])

    idx, mapping = _geom.unique_points(points, decimals=2)

    np.testing.assert_allclose(idx, expected_idx)
    np.testing.assert_allclose(mapping, expected_mapping)
