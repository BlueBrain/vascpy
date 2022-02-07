"""
Copyright (c) 2022 Blue Brain Project/EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np


def sort_points(points):
    """
    Returns a lexicographic sorting of the 3D coordinates first by x
    then by y and finally by z.
    Args:
        points: array[float, (N, 3)]

    Returns:
        sorted_indices: array[int, (N,)]
    """
    return np.lexsort((points[:, 2], points[:, 1], points[:, 0]))


def are_consecutive_pairs_different(points):
    """
    Given an array of points in lexicographic order it returns
    a mask where the i-th entry is True if it is identical with
    the entry i-1. By construction the first entry is always different.

    Example:

        points = [[0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1],
                  [0.1, 0.2, 0.3],
                  [0.1, 0.2, 0.3]]

        result: [True, False, True, False]
    """
    are_diff = np.empty(len(points), dtype=bool)
    are_diff[0] = True

    # check if the difference of the i-th and (i-1)th entries is zero
    are_close = np.isclose(points[1:] - points[:-1], 0.0)

    # if any coordinate xyz is not close, then the points are different
    are_diff[1:] = (~are_close).any(axis=1)
    return are_diff


def unique_points(points, decimals):
    """

    Args:
        points: array[float, (N, 3)]
            The array of points.
        decimals: int
            Points are rounded before sorting. This determines the number
            of significant digits.

    Returns:
        unique_indices: array[int, (M,)]
            The indices of the unique indices in the points array in the order that
            they appear in the array.
        inverse_mapping: array[int, (N,)]
            The mapping from the unique points back to the points array in the order
            they appear in the points array.

    Notes:

        inspired by numpy's unique1d
        https://docs.scipy.org/doc/numpy-1.3.x/reference/generated/numpy.unique1d.html

        Both unique and inverse mapping indices maintain the order of each point in the
        points array. Therefore this function is not the same as numpy's unique which does
        not maintain the order.
    """
    rounded_points = points.round(decimals=decimals)
    sorted_idx = sort_points(rounded_points)

    # check if each ordered points row is close to the one above
    is_unique = are_consecutive_pairs_different(rounded_points[sorted_idx])

    # consecutive duplicates has the same unique id
    duplicate_ids = np.cumsum(is_unique) - 1

    # maps the i-th row of points to the j-th row of sorted points
    to_sorted = np.argsort(sorted_idx)

    # maps each point in the initial array to the unique one
    inverse_mapping = np.empty_like(sorted_idx)

    # first point that appears in point array is the first indexed
    inverse_mapping[0] = 0

    # to keep track of the ids we visited so far
    # because the duplicat_ids is calculated on the sorted array
    # we need the to_sorted in order to go to the sorted indice
    visited = {duplicate_ids[to_sorted[0]]: 0}

    num_duplicates = 1
    for i in range(1, len(sorted_idx)):

        s_index = to_sorted[i]
        cid = duplicate_ids[s_index]

        # if our point is unique or if we encounter
        # the first of the duplicate points, a unique
        # index n is assigned
        if is_unique[s_index] or cid not in visited:

            inverse_mapping[i] = num_duplicates

            # keep track of the duplicate
            visited[cid] = num_duplicates
            num_duplicates += 1

        # the points is not unique and the first of the
        # duplicates is already registered
        else:

            # use the id of the first duplicate
            inverse_mapping[i] = visited[cid]

    # unique ids are sorted to maintain the order they appear
    # in the points array
    return np.sort(sorted_idx[is_unique]), inverse_mapping
