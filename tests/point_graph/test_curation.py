import numpy as np
import pandas as pd
from numpy import testing as npt
from pandas import testing as pdt

from vascpy.point_graph import curation as tested
from vascpy.point_vasculature import PointGraph

"""
def test_remove_vertices_from_edges():

    edges = np.array([[0, 1],
                      [1, 2],
                      [4, 5],
                      [7, 8]])

    to_remove_mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    new_edges = _c._remove_vertices_from_edges(to_remove_mask, edges)
    npt.assert_array_equal(new_edges, edges)

    to_remove_mask = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=bool)

    new_edges = _c._remove_vertices_from_edges(to_remove_mask, edges)
    npt.assert_array_equal(new_edges, [[0, 1], [1, 2], [3, 4], [5, 6]])

    to_remove_mask = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    new_edges = _c._remove_vertices_from_edges(to_remove_mask, edges)
    npt.assert_array_equal(new_edges, [[0, 1], [2, 3], [4, 5]])

    to_remove_mask = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    new_edges = _c._remove_vertices_from_edges(to_remove_mask, edges)
    npt.assert_array_equal(new_edges, [[0, 1], [2, 3]])

    to_remove_mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    new_edges = _c._remove_vertices_from_edges(to_remove_mask, edges)
    npt.assert_array_equal(new_edges, np.empty(shape=(0, 2), dtype=np.int64))

    to_remove_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    new_edges = _c._remove_vertices_from_edges(to_remove_mask, edges)
    npt.assert_array_equal(new_edges, np.empty(shape=(0, 2), dtype=np.int64))
"""


def test_edges_shorter_than():

    edges = np.array([[0, 1], [1, 2], [4, 5], [7, 8]])

    points = np.random.random((9, 3))

    points[4, :] *= 1000

    mask = tested._edges_shorter_than(points, edges, 10.0)

    npt.assert_array_equal(mask, [1, 1, 0, 1])


def test_edges_no_self_loops():

    edges = np.array([[0, 1], [2, 1], [3, 3]])

    mask = tested._edges_no_self_loops(edges)
    npt.assert_array_equal(mask, [1, 1, 0])


def test_curate_point_graph():

    points = np.random.random((12, 3))
    points[3] *= 100.0

    points = np.array(
        [
            [0.38, 0.26, 0.72],
            [0.14, 0.56, 0.94],
            [0.91, 0.33, 0.08],
            [11.65, 58.86, 90.05],
            [0.15, 0.82, 0.97],
            [0.88, 0.72, 0.04],
            [0.98, 0.81, 0.16],
            [0.66, 0.76, 0.74],
            [0.09, 0.49, 0.27],
            [0.00, 0.19, 0.23],
            [0.51, 0.48, 0.48],
            [0.11, 0.20, 0.01],
        ]
    )

    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 4], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10]]
    )

    point_graph = PointGraph.from_datasets(points, edges)

    tested.curate_point_graph(
        point_graph,
        remove_self_loops=True,
        remove_very_long_edges=True,
        remove_high_degree_vertices=True,
        remove_isolated_vertices=True,
    )

    # remove self loops will remove edge [4, 4]
    # remove bery long edges will remove [2, 3] and [3, 4]
    # remove high degree vertices will remove vertex 5 and all incident edges
    # remove isolated vertice will remove vertex 6, 7, 8, 9, 10, 11

    expected_node_properties = pd.DataFrame(
        np.array([[0.38, 0.26, 0.72], [0.14, 0.56, 0.94], [0.91, 0.33, 0.08]]),
        columns=["x", "y", "z"],
    )

    pdt.assert_frame_equal(point_graph.node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        np.array([[0, 1], [1, 2]]), columns=["start_node", "end_node"]
    )

    pdt.assert_frame_equal(point_graph.edge_properties, expected_edge_properties)
