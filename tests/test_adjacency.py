import numpy as np
import pytest
from numpy import testing as npt

from vascpy.utils.adjacency import AdjacencyMatrix, IncidenceMatrix


@pytest.fixture
def edges():
    """
                         e5
                 e4    |---| # self loop
                -----> 5<--|
      e0    e1 |   e2
    0 --> 1 --> 2 ---> 3 --> 4
               ^     ^   e3
               |     | e7
               ----- 6       7    8
                e6           isolated
    """
    #                  e0      e1      e2      e3      e4      e5      e6      e7
    return np.array([[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 5], [6, 2], [6, 3]])


@pytest.fixture
def adjacency_matrix(edges):
    return AdjacencyMatrix(edges, n_vertices=9)


@pytest.fixture
def incidence_matrix(edges):
    return IncidenceMatrix(edges, n_vertices=9)


def test_adjacency_matrix_n_edges(adjacency_matrix):
    assert adjacency_matrix.n_edges == 8


def test_adjacency_matrix_sparse_matrix(adjacency_matrix, edges):

    matrix = adjacency_matrix.as_sparse().toarray()

    expected = np.zeros((9, 9), dtype=np.int32)
    expected[edges[:, 0], edges[:, 1]] = 1

    npt.assert_array_equal(matrix, expected)


def test_adjacency_matrix__outdegrees(adjacency_matrix):
    npt.assert_array_equal(adjacency_matrix.outdegrees, [1, 1, 2, 1, 0, 1, 2, 0, 0])


def test_adjacency_matrix__indegreess(adjacency_matrix):
    npt.assert_array_equal(adjacency_matrix.indegrees, [0, 1, 2, 2, 1, 2, 0, 0, 0])


def test_adjacency_matrix__number_of_self_loops(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.number_of_self_loops(), 1)


def test_adjacency_matrix__degrees(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.degrees, [1, 2, 4, 3, 1, 3, 2, 0, 0])


def test_adjacency_matrix__sources(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.sources(), [0, 6])


def test_adjacency_matrix__sinks(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.sinks(), [4])


def test_adjacency_matrix__terminations(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.terminations(), [0, 4])


def test_adjacency_matrix__continuations(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.continuations(), [1])


def test_adjacency_matrix__isolated_vertices(adjacency_matrix):
    npt.assert_equal(adjacency_matrix.isolated_vertices(), [7, 8])


def test_adjacency_matrix__predecessors(adjacency_matrix):

    npt.assert_equal(adjacency_matrix.predecessors(0), [])
    npt.assert_equal(adjacency_matrix.predecessors(1), [0])
    npt.assert_equal(adjacency_matrix.predecessors(2), [1, 6])
    npt.assert_equal(adjacency_matrix.predecessors(3), [2, 6])
    npt.assert_equal(adjacency_matrix.predecessors(4), [3])
    npt.assert_equal(adjacency_matrix.predecessors(5), [2, 5])
    npt.assert_equal(adjacency_matrix.predecessors(6), [])
    npt.assert_equal(adjacency_matrix.predecessors(7), [])
    npt.assert_equal(adjacency_matrix.predecessors(8), [])


def test_adjacency_matrix__successors(adjacency_matrix):

    npt.assert_equal(adjacency_matrix.successors(0), [1])
    npt.assert_equal(adjacency_matrix.successors(1), [2])
    npt.assert_equal(adjacency_matrix.successors(2), [3, 5])
    npt.assert_equal(adjacency_matrix.successors(3), [4])
    npt.assert_equal(adjacency_matrix.successors(4), [])
    npt.assert_equal(adjacency_matrix.successors(5), [5])
    npt.assert_equal(adjacency_matrix.successors(6), [2, 3])
    npt.assert_equal(adjacency_matrix.successors(7), [])
    npt.assert_equal(adjacency_matrix.successors(8), [])


def test_adjacency_matrix__connected_components(adjacency_matrix):

    vertices, offsets = adjacency_matrix.connected_components()

    n_components = len(offsets) - 1

    assert n_components == 3

    first_component_vertices = vertices[offsets[0] : offsets[1]]
    second_component_vertices = vertices[offsets[1] : offsets[2]]
    third_component_vertices = vertices[offsets[2] : offsets[3]]

    npt.assert_equal(first_component_vertices, [0, 1, 2, 3, 4, 5, 6])
    npt.assert_equal(second_component_vertices, [7])
    npt.assert_equal(third_component_vertices, [8])


def test_adjacency_matrix_edge_ids(edges, adjacency_matrix):

    for i, (start_vertex, end_vertex) in enumerate(edges):
        assert adjacency_matrix.edge_index(start_vertex, end_vertex) == i

    assert adjacency_matrix.edge_index(0, 4) == -1
    assert adjacency_matrix.edge_index(1, 0) == -1
    assert adjacency_matrix.edge_index(2, 1) == -1
    assert adjacency_matrix.edge_index(3, 2) == -1
    assert adjacency_matrix.edge_index(1, 1) == -1


def test_incidence_matrix__incident(incidence_matrix):

    npt.assert_equal(incidence_matrix.incident(0), [0])
    npt.assert_equal(incidence_matrix.incident(1), [0, 1])
    npt.assert_equal(incidence_matrix.incident(2), [1, 2, 4, 6])
    npt.assert_equal(incidence_matrix.incident(3), [2, 3, 7])
    npt.assert_equal(incidence_matrix.incident(4), [3])
    npt.assert_equal(incidence_matrix.incident(5), [4, 5])
    npt.assert_equal(incidence_matrix.incident(6), [6, 7])
    npt.assert_equal(incidence_matrix.incident(7), [])
    npt.assert_equal(incidence_matrix.incident(8), [])
