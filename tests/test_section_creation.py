import os
from collections import deque

import numpy as np
from numpy import testing as npt

from vascpy.utils import section_creation as tested
from vascpy.utils.section_creation import create_chains

_path = os.path.dirname(os.path.abspath(__file__))


def _edges_and_doc(edges, graph_string):
    return "Input Edges:\n\n{}\n\nGraph:\n\n{}".format(edges, graph_string)


def _print_section_list(sections):

    string = ""

    for i, section in enumerate(sections):

        str_section = "{0} : ".format(i)

        if len(section) > 0:

            str_section += "["
            for edge in section:
                str_section += "({0}, {1}), ".format(*edge)

            str_section += "]"

        else:

            str_section += "[]"

        string += str_section + "\n"

    return string


def _print_structure_comparison(sections_actual, sections_algorithm, include_str=""):

    str_actual = "\n\nsections :\n--------\n" + _print_section_list(sections_actual)
    str_algori = "\n\nalgorithm:\n--------\n" + _print_section_list(sections_algorithm)

    return "\n\n" + include_str + "\n" + str_actual + str_algori


def _print_connectivity_comparison(secs_connec_actual, secs_connec_algorithm, include_str=""):

    str_actual = "\n".join(str(el) for el in secs_connec_actual)
    str_algori = "\n".join(str(el) for el in secs_connec_algorithm)
    return (
        "\n\n"
        + include_str
        + "\n\nconnectivity:\n--------\n{0}\n\nalgorithm:\n--------\n{1}\n".format(
            str_actual, str_algori
        )
    )


def assert_chains_equal(first, second, include_string=""):

    # check length
    assert len(first) == len(second), _print_structure_comparison(first, second, include_string)

    if len(first) > 0 and len(second) > 0:
        for first_section, second_section in zip(first, second):

            assert len(first_section) == len(second_section), _print_structure_comparison(
                first, second, include_string
            )
            assert np.equal(first_section, second_section).all(), _print_structure_comparison(
                first, second, include_string
            )


def assert_chain_connectivity_equal(first, second, include_string=""):

    if len(first) > 0 and len(second) > 0:
        assert np.all(np.asarray(first) == np.asarray(second)), _print_connectivity_comparison(
            first, second, include_string
        )


def test_create_chains_simple_graph():

    edges = np.array(
        [
            [22, 21],
            [21, 19],
            [23, 24],
            [24, 25],
            [25, 20],
            [20, 19],
            [19, 18],
            [18, 17],
            [17, 16],
            [16, 15],
            [15, 14],
            [14, 0],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [100, 101],
            [101, 102],
            [102, 103],
            [103, 104],
            [102, 105],
            [105, 106],
        ],
        dtype=np.int64,
    )

    secs_target = [
        ((4, 3), (3, 2), (2, 1), (1, 0)),
        ((0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)),
        ((0, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)),
        ((19, 20), (20, 25), (25, 24), (24, 23)),
        ((19, 21), (21, 22)),
        ((100, 101), (101, 102)),
        ((102, 103), (103, 104)),
        ((102, 105), (105, 106)),
    ]

    secs_connec_target = np.array([[0, 1], [0, 2], [2, 3], [2, 4], [5, 6], [5, 7]], dtype=np.int64)

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)

    secs, secs_connec = create_chains(edges, edges.max() + 1, return_index=False)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_create_chains_cyclic_graph():

    edges = np.array(
        ([0, 1], [1, 2], [2, 3], [3, 4], [4, 7], [2, 5], [5, 6], [6, 7], [7, 8], [8, 9])
    )

    secs_target = [
        deque(((0, 1), (1, 2))),
        deque(((2, 3), (3, 4), (4, 7))),
        deque(((2, 5), (5, 6), (6, 7))),
        deque(((7, 8), (8, 9))),
    ]

    secs_connec_target = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int64)

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_create_chains_single_cycle_graph():

    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int64)

    secs_target = [[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]]
    secs_connec_target = []

    # -----------------------------------------------------------------------------------

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_create_chains_two_cycles():
    r"""
    0 - 1 - 2   6 - 7 - 8
    |       |    \     /
    5 - 4 - 3    10 - 9
    """
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [6, 7], [7, 8], [8, 9], [9, 10], [10, 6]],
        dtype=np.int64,
    )

    secs_target = [
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        [(6, 7), (7, 8), (8, 9), (9, 10), (10, 6)],
    ]

    secs_connec_target = []

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    msg = _edges_and_doc(edges, test_create_chains_two_cycles.__doc__)
    assert_chains_equal(secs_target, secs, include_string=msg)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec, include_string=msg)


def test_create_chains__two_cycles_linked():
    r"""
    0 - 1 - 2 - 6 - 7 - 8
    |       |    \     /
    5 - 4 - 3    10 - 9
    """
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 6],
            [2, 6],
        ],
        dtype=np.int64,
    )

    secs_target = [
        [(2, 1), (1, 0), (0, 5), (5, 4), (4, 3), (3, 2)],
        [(2, 6)],
        [(6, 7), (7, 8), (8, 9), (9, 10), (10, 6)],
    ]

    secs_connec_target = []

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    msg = _edges_and_doc(edges, test_create_chains__two_cycles_linked.__doc__)
    assert_chains_equal(secs_target, secs, include_string=msg)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec, include_string=msg)


def test_capillary_loop():

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 13],
            [3, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
        ]
    )

    r"""
                  4 - 5 - 6 - 7  
                 /             \
    0 - 1 - 2 - 3               13 - 14 - 15 - 16
                 \             /
                  8    10    12
                   \  /  \  /
                     9    11
    """

    secs_target = [
        [(0, 1), (1, 2), (2, 3)],
        [(3, 4), (4, 5), (5, 6), (6, 7), (7, 13)],
        [(3, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)],
        [(13, 14), (14, 15), (15, 16)],
    ]

    secs_connec_target = [[0, 1], [0, 2], [1, 3], [2, 3]]

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    msg = _edges_and_doc(edges, test_capillary_loop.__doc__)
    assert_chains_equal(secs_target, secs, include_string=msg)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec, include_string=msg)


def test_long_capillary_loop():

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 19],
            (3, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            [19, 20],
            [20, 21],
            [21, 22],
        ]
    )

    r"""
                  4 - 5 - 6 - 7  
                 /             \
    0 - 1 - 2 - 3               19 - 20 - 21 - 22
                |              /
                8            18
                |             |
                9            17
                |             |
               10            16
                |             |
               11            15
                \            /
                 12 - 13 - 14 
    """

    secs_target = [
        [(0, 1), (1, 2), (2, 3)],
        [(3, 4), (4, 5), (5, 6), (6, 7), (7, 19)],
        [
            (3, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
        ],
        [(19, 20), (20, 21), (21, 22)],
    ]

    secs_connec_target = [[0, 1], [0, 2], [1, 3], [2, 3]]

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    msg = _edges_and_doc(edges, test_long_capillary_loop.__doc__)
    assert_chains_equal(secs_target, secs, include_string=msg)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec, include_string=msg)


def test_self_referential_node():
    r"""
                  5 - 9
                 /             
    0 - 1 - 2 - 3      4-4
                 \
                  6 - 10
    """

    edges = np.array([[1, 0], [1, 2], [2, 3], [4, 4], [3, 5], [5, 9], [3, 6], [6, 10]])

    secs_target = [[(0, 1), (1, 2), (2, 3)], [(3, 5), (5, 9)], [(3, 6), (6, 10)]]

    secs_connec_target = [[0, 1], [0, 2]]

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    msg = _edges_and_doc(edges, test_self_referential_node.__doc__)
    assert_chains_equal(secs_target, secs, include_string=msg)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec, include_string=msg)


def test_capillary_loop_bidrectional_edge():
    r"""
                  4 - 5 - 6 - 7  
                 /             \
    0 - 1 - 2 - 3               13 - 14 - 15 - 16
                 \             /
                  8    10    12
                   \  /  \  /
                     9    11
    """

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 13],
            [13, 7],
            [3, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
        ]
    )

    secs_target = [
        [(0, 1), (1, 2), (2, 3)],
        [(3, 4), (4, 5), (5, 6), (6, 7), (7, 13)],
        [(3, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)],
        [(13, 14), (14, 15), (15, 16)],
    ]

    secs_connec_target = [[0, 1], [0, 2], [1, 3], [2, 3]]

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    msg = _edges_and_doc(edges, test_capillary_loop_bidrectional_edge.__doc__)
    assert_chains_equal(secs_target, secs, include_string=msg)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec, include_string=msg)


def test_triangle_loop():

    edges = np.array([[0, 1], [1, 2], [2, 0]])

    secs_target = [[(0, 1), (1, 2), (2, 0)]]
    secs_connec_target = []

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_square_loop():

    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    secs_target = [[(0, 1), (1, 2), (2, 3), (3, 0)]]
    secs_connec_target = []

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_single_edge():

    edges = np.array([[0, 1]])

    secs_target = [[(0, 1)]]
    secs_connec_target = []

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_single_edge_fork():

    edges = np.array([(0, 1), (1, 2), (1, 3)])

    secs_target = [[(0, 1)], [(1, 2)], [(1, 3)]]
    secs_connec_target = [[0, 1], [0, 2]]

    secs, secs_connec, index_secs = create_chains(edges, edges.max() + 1, return_index=True)

    assert_chains_equal(secs_target, secs)
    assert_chain_connectivity_equal(secs_connec_target, secs_connec)


def test_is_flow_through():

    in_out_degrees_valid_cases = [[1, 1], [0, 1], [1, 0], [0, 2], [2, 0]]

    for indeg, outdeg in in_out_degrees_valid_cases:
        assert tested._is_flow_through(indeg, outdeg)

    in_out_degrees_invalid_cases = [[1, 2], [2, 1], [1, 3], [3, 1], [2, 2], [0, 3], [3, 0], [3, 3]]

    for indeg, outdeg in in_out_degrees_invalid_cases:
        assert not tested._is_flow_through(indeg, outdeg)


def test_add_to_section():

    section = deque()

    tested._add_to_section(section, 0, 1)
    assert list(section) == [(0, 1)]

    tested._add_to_section(section, 1, 2)
    assert list(section) == [(0, 1), (1, 2)]

    tested._add_to_section(section, 0, 3)
    assert list(section) == [(3, 0), (0, 1), (1, 2)]

    tested._add_to_section(section, 4, 3)
    assert list(section) == [(4, 3), (3, 0), (0, 1), (1, 2)]

    tested._add_to_section(section, 5, 2)
    assert list(section) == [(4, 3), (3, 0), (0, 1), (1, 2), (2, 5)]

    # unconnected, nothing happens
    tested._add_to_section(section, 10, 11)
    assert list(section) == [(4, 3), (3, 0), (0, 1), (1, 2), (2, 5)]


def test_chain_connectivity():

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 13],
            [7, 13],
            [3, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
        ]
    )

    chains = [
        [(0, 1), (1, 2), (2, 3)],
        [(3, 4), (4, 5), (5, 6), (6, 7), (7, 13)],
        [(3, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)],
        [(13, 14), (14, 15), (15, 16)],
    ]

    expected_connectivity = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])

    npt.assert_array_equal(tested._chain_connectivity(edges, chains), expected_connectivity)


def test_map_chains_to_original_edges():

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 13],
            [7, 13],
            [3, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
        ]
    )

    chains = [
        [(0, 1), (1, 2), (2, 3)],
        [(3, 4), (4, 5), (5, 6), (6, 7), (7, 13)],
        [(3, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)],
        [(13, 14), (14, 15), (15, 16)],
    ]

    expected = [
        np.array([0, 1, 3]),
        np.array([4, 5, 6, 7, 9]),
        np.array([10, 11, 12, 13, 14, 15]),
        np.array([16, 17, 18]),
    ]

    values = tested._map_chains_to_original_edges(edges, chains)
    assert_chains_equal(values, expected)
