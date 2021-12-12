import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pandas import testing as pdt

from vascpy import specs
from vascpy.exceptions import VasculatureAPIError
from vascpy.point_vasculature import PointVasculature

DATAPATH = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def node_properties():
    return pd.DataFrame(
        {
            "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
            "y": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
            "z": np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
            "diameter": np.array(
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32
            ),
            "property1": np.array(
                [3.0, 1.0, 4.0, 1.0, 4.0, 5.0, 6.0, 8.0, 9.0, 0.0], dtype=np.float64
            ),
        },
        index=pd.RangeIndex(start=0, stop=10, step=1),
        columns=["x", "y", "z", "diameter", "property1"],
    )


@pytest.fixture
def edge_properties():
    return pd.DataFrame(
        {
            "start_node": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
            "end_node": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
            "type": np.array([1, 1, 1, 0, 0, 2, 5, 4, 4], dtype=np.int32),
            "section_id": np.array([0, 0, 0, 1, 1, 2, 3, 4, 4], dtype=np.int32),
            "segment_id": np.array([0, 1, 2, 0, 1, 0, 0, 0, 1], dtype=np.int32),
            "property1": np.array([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8),
        },
        index=pd.RangeIndex(start=0, stop=9, step=1),
        columns=["start_node", "end_node", "type", "section_id", "segment_id", "property1"],
    )


@pytest.fixture
def degrees():
    return np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 1])


@pytest.fixture
def points(node_properties):
    return node_properties.loc[:, ["x", "y", "z"]].to_numpy()


@pytest.fixture
def diameters(node_properties):
    return node_properties.loc[:, "diameter"].to_numpy()


@pytest.fixture
def edges(edge_properties):
    return edge_properties.loc[:, ["start_node", "end_node"]].to_numpy()


@pytest.fixture
def edge_types(edge_properties):
    return edge_properties.loc[:, "type"].to_numpy()


@pytest.fixture
def point_vasculature(node_properties, edge_properties):
    return PointVasculature(node_properties, edge_properties)


def test_constructor(point_vasculature, node_properties, edge_properties):

    pdt.assert_frame_equal(node_properties, point_vasculature._node_properties)
    pdt.assert_frame_equal(edge_properties, point_vasculature._edge_properties)

    # just execute loading to check if it works. Values are checked in other tests
    PointVasculature.load(DATAPATH / "h5/fork_diameters_point_spec.h5")


def test_from_datasets(point_vasculature):

    p = PointVasculature.from_datasets(
        point_vasculature.points,
        point_vasculature.edges,
        point_data={
            "diameter": point_vasculature.diameters,
            "property1": point_vasculature.node_properties["property1"].to_numpy(),
        },
        edge_data={
            "type": point_vasculature.edge_types,
            "section_id": point_vasculature.edge_properties["section_id"].to_numpy(),
            "segment_id": point_vasculature.edge_properties["segment_id"].to_numpy(),
            "property1": point_vasculature.edge_properties["property1"].to_numpy(),
        },
    )

    pdt.assert_frame_equal(p.node_properties, point_vasculature.node_properties)
    pdt.assert_frame_equal(p.edge_properties, point_vasculature.edge_properties)


def test_node_validation(node_properties, edge_properties):

    PointVasculature._validate_node_properties(node_properties)

    del node_properties["x"]
    with pytest.raises(VasculatureAPIError):
        PointVasculature._validate_node_properties(node_properties)

    del edge_properties["type"]
    with pytest.raises(VasculatureAPIError):
        PointVasculature._validate_edge_properties(edge_properties)


def test_get_properties(point_vasculature, points, edges, edge_types, diameters):

    assert point_vasculature.n_nodes == 10
    assert point_vasculature.n_edges == 9

    npt.assert_allclose(points, point_vasculature.points)
    npt.assert_array_equal(point_vasculature.edges, edges)
    npt.assert_array_equal(point_vasculature.edge_types, edge_types)
    npt.assert_allclose(point_vasculature.diameters, diameters)

    beg_points, end_points = point_vasculature.segment_points
    npt.assert_allclose(beg_points, points[edges[:, 0]])
    npt.assert_allclose(end_points, points[edges[:, 1]])

    beg_diameters, end_diameters = point_vasculature.segment_diameters
    npt.assert_allclose(beg_diameters, diameters[edges[:, 0]])
    npt.assert_allclose(end_diameters, diameters[edges[:, 1]])

    values = point_vasculature.node_properties.loc[[0, 2, 7, 9], "property1"]
    npt.assert_allclose(values, [3.0, 4.0, 8.0, 0.0])

    values = point_vasculature.node_properties.loc[[1, 5], ["x", "z"]]
    npt.assert_allclose(values, [[1.0, 3.0], [5.0, 7.0]])

    index = point_vasculature.edge_properties.index

    values = point_vasculature.edge_properties.loc[[8, 5], "property1"]
    npt.assert_allclose(point_vasculature.edge_properties.loc[[8, 5], "property1"], [1, 4])

    values = point_vasculature.edge_properties.loc[[0, 1, 2], ["start_node", "end_node"]]
    npt.assert_allclose(values, [[0, 1], [1, 2], [2, 3]])


def test_set_properties(point_vasculature):

    new_points = np.random.random((10, 3))
    point_vasculature.points = new_points
    npt.assert_allclose(point_vasculature.points, new_points)

    new_edges = np.random.randint(0, 8, size=(9, 2))
    point_vasculature.edges = new_edges
    npt.assert_array_equal(point_vasculature.edges, new_edges)

    new_edge_types = np.random.randint(0, 100, 9)
    point_vasculature.edge_types = new_edge_types
    npt.assert_array_equal(point_vasculature.edge_types, new_edge_types)

    new_diameters = np.random.random(10)
    point_vasculature.diameters = new_diameters
    npt.assert_allclose(point_vasculature.diameters, new_diameters)

    old_points = point_vasculature.points
    new_values = np.array([[9, 10], [10, 11], [11, 12]])
    point_vasculature.node_properties.loc[[1, 3, 5], ["x", "z"]] = new_values

    old_points[(1, 3, 5), 0] = new_values[:, 0]
    old_points[(1, 3, 5), 2] = new_values[:, 1]
    npt.assert_allclose(point_vasculature.points, old_points)

    new_values = np.random.random(10)
    point_vasculature.node_properties.loc[:, "property1"] = new_values
    npt.assert_allclose(point_vasculature.node_properties.loc[:, "property1"], new_values)

    old_edges = point_vasculature.edges
    new_values = np.array([[1, 3], [4, 1]])

    point_vasculature.edge_properties.loc[[0, 7], ["start_node", "end_node"]] = new_values

    old_edges[[0, 7], 0] = new_values[:, 0]
    old_edges[[0, 7], 1] = new_values[:, 1]

    npt.assert_allclose(point_vasculature.edges, old_edges)

    new_values = np.random.random(9)
    point_vasculature.edge_properties.loc[:, "property1"] = new_values
    npt.assert_allclose(point_vasculature.edge_properties.loc[:, "property1"], new_values)


def test_measurements(point_vasculature):

    npt.assert_allclose(point_vasculature.length, 15.588457)
    npt.assert_allclose(point_vasculature.area, 382.2921)
    npt.assert_allclose(point_vasculature.volume, 771.3182)


def test_topology(point_vasculature, degrees):

    adjacency = point_vasculature.adjacency_matrix
    res_degrees = adjacency.degrees

    npt.assert_array_equal(res_degrees, degrees)
    npt.assert_array_equal(res_degrees, point_vasculature.degrees)


def test_sonata_io(point_vasculature):

    SPEC = specs.SpecSONATA

    with NamedTemporaryFile(suffix=".h5") as tfile:
        filename = tfile.name

        # load regular morphology file into PointVasculature
        v1 = point_vasculature
        v1.save_sonata(filename)

        # load sonata node population file into PointVasculature
        v2 = PointVasculature.load_sonata(filename)

        for prop, dtype in SPEC.SONATA_POINT_DTYPES.items():

            v1_values = v1.node_properties.loc[:, prop].to_numpy()
            v2_values = v2.node_properties.loc[:, prop].to_numpy()
            npt.assert_allclose(v1_values, v2_values)
            assert v2_values.dtype == dtype

        for prop, dtype in SPEC.SONATA_EDGE_DTYPES.items():

            v1_values = v1.edge_properties.loc[:, prop].to_numpy()
            v2_values = v2.edge_properties.loc[:, prop].to_numpy()
            npt.assert_allclose(v1_values, v2_values)

            # we cannot use uint64 due to the funky promotion rules of numpy
            if prop in {"start_node", "end_node"}:
                assert v2_values.dtype == np.int64
            else:
                assert v2_values.dtype == dtype


def test_remove_nodes(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]

    We remove nodes [1, 5, 7] and any edges pointing to them:

    Nodes : 0 - 2 3 4 - 6 - 8 9 10
    Edges : -----, -----, [2, 3], [3, 4], -----, -----, -----, -----, [8, 9]

    Removing however the edges corresponding to the nodes to be removed, leaves some other
    nodes without an edge. These nodes need to be removed because the vasculature spec
    does not support isolated points.

    Nodes : - - 2 3 4 - - - 8 9
    Edges : -----, -----, [2, 3], [3, 4], -----, -----, -----, -----, [8, 9]

    which will be reindexed into:

    Nodes : - - 0 1 2 - - - 3 4
    Edges : -----, -----, [0, 1], [1, 2], -----, -----, -----, -----, [3, 4]
    """
    expected_node_properties = point_vasculature.node_properties.iloc[[2, 3, 4, 8, 9], :]

    point_vasculature.remove(node_indices=[1, 5, 7])

    pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        {
            "start_node": np.array([0, 1, 3], dtype=np.int64),
            "end_node": np.array([1, 2, 4], dtype=np.int64),
            "type": np.array([1, 0, 4], dtype=np.int32),
            "section_id": np.array([0, 1, 4], dtype=np.int32),
            "segment_id": np.array([2, 0, 1], dtype=np.int32),
            "property1": np.array([7, 6, 1], dtype=np.int8),
        },
        index=np.array([2, 3, 8], dtype=np.int64),
        columns=["start_node", "end_node", "type", "section_id", "segment_id", "property1"],
    )
    pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)

    # now lets' try read and write it. As custom properties are not part of the sonata spec they
    # will be discarded
    expected_node_properties = expected_node_properties.drop(columns=["property1"])
    expected_edge_properties = expected_edge_properties.drop(columns=["property1"])

    # Furthermore sonata supports only an incremental and contiguous index
    expected_node_properties.reset_index(inplace=True, drop=True)
    expected_edge_properties.reset_index(inplace=True, drop=True)

    with NamedTemporaryFile(suffix=".h5") as tfile:
        filename = tfile.name
        point_vasculature.save_sonata(filename)
        point_vasculature2 = PointVasculature.load_sonata(filename)

    # as sonata stores specific dtypes, it is expected to get dtype missmatches given that
    # the test data does to respect the sonata spec dtypes
    pdt.assert_frame_equal(
        point_vasculature2.node_properties, expected_node_properties, check_dtype=False
    )
    pdt.assert_frame_equal(
        point_vasculature2.edge_properties, expected_edge_properties, check_dtype=False
    )


def test_remove_edges(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]

    We remove edges [0, 1, 4, 7, 8] and any nodes that are left unreferenced:

    Nodes : - - 2 3 4 5 6 7 - - -
    Edges : -----, -----, [2, 3], [3, 4], -----, [5, 6], [6, 7], -----, -----

    which will be reindexed into:

    Nodes : - - 0 1 2 3 4 5 - - -
    Edges : -----, -----, [0, 1], [1, 2], -----, [3, 4], [4, 5], -----, -----
    """
    expected_node_properties = point_vasculature.node_properties.iloc[[2, 3, 4, 5, 6, 7], :]

    point_vasculature.remove(edge_indices=[0, 1, 4, 7, 8])

    pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        {
            "start_node": np.array([0, 1, 3, 4], dtype=np.int64),
            "end_node": np.array([1, 2, 4, 5], dtype=np.int64),
            "type": np.array([1, 0, 2, 5], dtype=np.int32),
            "section_id": np.array([0, 1, 2, 3], dtype=np.int32),
            "segment_id": np.array([2, 0, 0, 0], dtype=np.int32),
            "property1": np.array([7, 6, 4, 3], dtype=np.int8),
        },
        index=np.array([2, 3, 5, 6], dtype=np.int64),
        columns=["start_node", "end_node", "type", "section_id", "segment_id", "property1"],
    )
    pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)

    # now lets' try read and write it. As custom properties are not part of the sonata spec they
    # will be discarded
    expected_node_properties = expected_node_properties.drop(columns=["property1"])
    expected_edge_properties = expected_edge_properties.drop(columns=["property1"])

    # Furthermore sonata supports only an incremental and contiguous index
    expected_node_properties.reset_index(inplace=True, drop=True)
    expected_edge_properties.reset_index(inplace=True, drop=True)

    with NamedTemporaryFile(suffix=".h5") as tfile:
        filename = tfile.name
        point_vasculature.save_sonata(filename)
        point_vasculature2 = PointVasculature.load_sonata(filename)

    # as sonata stores specific dtypes, it is expected to get dtype missmatches given that
    # the test data does to respect the sonata spec dtypes
    pdt.assert_frame_equal(
        point_vasculature2.node_properties, expected_node_properties, check_dtype=False
    )
    pdt.assert_frame_equal(
        point_vasculature2.edge_properties, expected_edge_properties, check_dtype=False
    )


def test_remove__nodes_from_the_end(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]
    """
    expected_node_properties = point_vasculature.node_properties.copy()
    expected_edge_properties = point_vasculature.edge_properties.copy()

    for node_id in [9, 8, 7, 6, 5, 4, 3, 2]:

        expected_node_properties = expected_node_properties.iloc[:-1]
        expected_edge_properties = expected_edge_properties.iloc[:-1]

        point_vasculature.remove(node_indices=[node_id])
        pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)
        pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)


def test_remove__nodes_from_the_start(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]
    """
    expected_node_properties = point_vasculature.node_properties.copy()
    expected_edge_properties = point_vasculature.edge_properties.copy()

    for node_id in [0] * 8:

        expected_node_properties = expected_node_properties.iloc[1:]
        expected_edge_properties = expected_edge_properties.iloc[1:]
        expected_edge_properties.loc[:, ["start_node", "end_node"]] -= 1

        point_vasculature.remove(node_indices=[node_id])
        pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)
        pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)


def test_remove__edges_from_the_end(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]
    """
    expected_node_properties = point_vasculature.node_properties.copy()
    expected_edge_properties = point_vasculature.edge_properties.copy()

    for edge_id in [8, 7, 6, 5, 4, 3, 2, 1]:

        expected_node_properties = expected_node_properties.iloc[:-1]
        expected_edge_properties = expected_edge_properties.iloc[:-1]

        point_vasculature.remove(edge_indices=[edge_id])

        pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)
        pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)


def test_remove__edges_from_the_start(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]
    """
    expected_node_properties = point_vasculature.node_properties.copy()
    expected_edge_properties = point_vasculature.edge_properties.copy()

    for edge_id in [0] * 8:

        expected_node_properties = expected_node_properties.iloc[1:]
        expected_edge_properties = expected_edge_properties.iloc[1:]
        expected_edge_properties.loc[:, ["start_node", "end_node"]] -= 1

        point_vasculature.remove(edge_indices=[edge_id])

        pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)
        pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)


def test_remove_nodes_and_edges(point_vasculature):
    """
    Nodes : 0 1 2 3 4 5 6 7 8 9
    Edges : [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]

    We remove nodes [0, 5, 9] and edges [[1, 2], [8, 9], [3, 4]]. Removing also any
    not referenced nodes results in:

    Nodes : - - 2 3 - - 6 7 8 -
    Edges : [0, 1], -----, [2, 3], -----, -----, -----, [6, 7], [7, 8], -----

    which will be reindexed into:

    Nodes : - - 0 1 - - 2 3 4 -
    Edges : -----, -----, [0, 1], -----, -----, -----, [2, 3], [3, 4], -----
    """
    expected_node_properties = point_vasculature.node_properties.iloc[[2, 3, 6, 7, 8], :]

    point_vasculature.remove(node_indices=[0, 5, 9], edge_indices=[1, 3, 8])

    pdt.assert_frame_equal(point_vasculature.node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        {
            "start_node": np.array([0, 2, 3], dtype=np.int64),
            "end_node": np.array([1, 3, 4], dtype=np.int64),
            "type": np.array([1, 5, 4], dtype=np.int32),
            "section_id": np.array([0, 3, 4], dtype=np.int32),
            "segment_id": np.array([2, 0, 0], dtype=np.int32),
            "property1": np.array([7, 3, 2], dtype=np.int8),
        },
        index=np.array([2, 6, 7], dtype=np.int64),
        columns=["start_node", "end_node", "type", "section_id", "segment_id", "property1"],
    )
    pdt.assert_frame_equal(point_vasculature.edge_properties, expected_edge_properties)

    # now lets' try read and write it. As custom properties are not part of the sonata spec they
    # will be discarded
    expected_node_properties = expected_node_properties.drop(columns=["property1"])
    expected_edge_properties = expected_edge_properties.drop(columns=["property1"])

    # Furthermore sonata supports only an incremental and contiguous index
    expected_node_properties.reset_index(inplace=True, drop=True)
    expected_edge_properties.reset_index(inplace=True, drop=True)

    with NamedTemporaryFile(suffix=".h5") as tfile:
        filename = tfile.name
        point_vasculature.save_sonata(filename)
        point_vasculature2 = PointVasculature.load_sonata(filename)

    # as sonata stores specific dtypes, it is expected to get dtype missmatches given that
    # the test data does to respect the sonata spec dtypes
    pdt.assert_frame_equal(
        point_vasculature2.node_properties, expected_node_properties, check_dtype=False
    )
    pdt.assert_frame_equal(
        point_vasculature2.edge_properties, expected_edge_properties, check_dtype=False
    )


def test_file_formats(point_vasculature):

    with NamedTemporaryFile(suffix=".h5") as tfile:

        point_vasculature.save(tfile.name)
        point_vasculature_2 = point_vasculature.load(tfile.name)

    with NamedTemporaryFile(suffix=".vtk") as tfile:

        point_vasculature_2.save(tfile.name)
        point_vasculature_3 = point_vasculature_2.load(tfile.name)

    pdt.assert_frame_equal(point_vasculature.node_properties, point_vasculature_2.node_properties)
    pdt.assert_frame_equal(point_vasculature.edge_properties, point_vasculature_2.edge_properties)

    pdt.assert_frame_equal(point_vasculature.node_properties, point_vasculature_3.node_properties)
    pdt.assert_frame_equal(point_vasculature.edge_properties, point_vasculature_3.edge_properties)

    with pytest.raises(VasculatureAPIError):
        point_vasculature_3.save("test.ogg")

    with pytest.raises(VasculatureAPIError):
        PointVasculature.load("test.ogg")


def test_conversion_invariant(point_vasculature):

    point_vasculature_2 = point_vasculature.as_section_graph().as_point_graph()

    # section representation does not support extra properties
    point_vasculature.node_properties.drop(columns="property1", inplace=True)
    point_vasculature.edge_properties.drop(columns="property1", inplace=True)

    pdt.assert_frame_equal(point_vasculature_2.node_properties, point_vasculature.node_properties)
    pdt.assert_frame_equal(point_vasculature_2.edge_properties, point_vasculature.edge_properties)
