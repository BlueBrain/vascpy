import pathlib
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pandas import testing as pdt

from vascpy.point_graph import io as tested
from vascpy.point_vasculature import PointGraph

DATAPATH = pathlib.Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def module_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp(Path(__file__).stem)


@pytest.fixture(scope="session")
def single_fork():

    points = np.array(
        [
            [1332.100, 443.800, 1911.000],
            [1331.610, 443.863, 1910.780],
            [1332.140, 443.581, 1910.990],
            [1332.550, 444.087, 1911.090],
        ],
        dtype=np.float32,
    )

    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)

    return {"points": points, "edges": edges}


def test_hdf5_read_write__single_fork(single_fork, module_tmpdir):

    path = module_tmpdir.join("single_fork_hdf5_1.h5")

    with h5py.File(path, "w") as fd:

        fd.create_dataset("points", data=single_fork["points"])
        fd.create_dataset("edges", data=single_fork["edges"])
        # group without datasets
        pgroup = fd.create_group("point_properties")
        egroup = fd.create_group("edge_properties")

    node_properties, edge_properties = tested.HDF5.read(path)

    expected_node_properties = pd.DataFrame(
        single_fork["points"], columns=["x", "y", "z"], dtype=np.float32
    )

    pdt.assert_frame_equal(node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        single_fork["edges"], columns=["start_node", "end_node"], dtype=np.int64
    )

    pdt.assert_frame_equal(edge_properties, expected_edge_properties)

    path = module_tmpdir.join("single_fork_hdf5_2.h5")

    tested.HDF5.write(path, node_properties, edge_properties)
    node_properties, edge_properties = tested.HDF5.read(path)

    pdt.assert_frame_equal(node_properties, expected_node_properties)
    pdt.assert_frame_equal(edge_properties, expected_edge_properties)


@pytest.fixture(scope="session")
def single_fork_props():

    points = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.7, 0.8], [0.9, 1.0, 1.1]], dtype=np.float32
    )

    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)

    point_properties = {
        "diameter": np.array([4.123, 4.123, 3.36162, 3.96078], dtype=np.float32),
        "point_property1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "point_property2": np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64),
    }

    edge_properties = {
        "type": np.array([0, 0, 0], dtype=np.int8),
        "edge_property1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "edge_property2": np.array([0.4, 0.5, 0.6], dtype=np.float32),
    }

    return {
        "points": points,
        "edges": edges,
        "point_data": point_properties,
        "edge_data": edge_properties,
    }


def test_hdf5_read_write__fork_multiple_properties(single_fork_props, module_tmpdir):

    path = module_tmpdir.join("single_fork_props_hdf5_1.h5")
    data = single_fork_props

    with h5py.File(path, "w") as fd:

        fd.create_dataset("points", data=data["points"])
        fd.create_dataset("edges", data=data["edges"])

        pgroup = fd.create_group("point_properties", track_order=True)
        pgroup.create_dataset("diameter", data=data["point_data"]["diameter"])
        pgroup.create_dataset("point_property1", data=data["point_data"]["point_property1"])
        pgroup.create_dataset("point_property2", data=data["point_data"]["point_property2"])

        egroup = fd.create_group("edge_properties", track_order=True)
        egroup.create_dataset("type", data=data["edge_data"]["type"])
        egroup.create_dataset("edge_property1", data=data["edge_data"]["edge_property1"])
        egroup.create_dataset("edge_property2", data=data["edge_data"]["edge_property2"])

    node_properties, edge_properties = tested.HDF5.read(path)

    expected_node_properties = pd.DataFrame(
        {
            "x": single_fork_props["points"][:, 0],
            "y": single_fork_props["points"][:, 1],
            "z": single_fork_props["points"][:, 2],
            "diameter": single_fork_props["point_data"]["diameter"],
            "point_property1": single_fork_props["point_data"]["point_property1"],
            "point_property2": single_fork_props["point_data"]["point_property2"],
        },
        columns=["x", "y", "z", "diameter", "point_property1", "point_property2"],
    )

    pdt.assert_frame_equal(node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        {
            "start_node": single_fork_props["edges"][:, 0],
            "end_node": single_fork_props["edges"][:, 1],
            "type": single_fork_props["edge_data"]["type"],
            "edge_property1": single_fork_props["edge_data"]["edge_property1"],
            "edge_property2": single_fork_props["edge_data"]["edge_property2"],
        }
    )

    pdt.assert_frame_equal(edge_properties, expected_edge_properties)

    path = module_tmpdir.join("single_fork_props_hdf5_2.h5")

    tested.HDF5.write(path, node_properties, edge_properties)
    node_properties, edge_properties = tested.HDF5.read(path)

    pdt.assert_frame_equal(node_properties, expected_node_properties)
    pdt.assert_frame_equal(edge_properties, expected_edge_properties)


def test_vtk_read_write__single_fork(single_fork, module_tmpdir):

    path = module_tmpdir.join("single_fork_vtk_1.vtk")

    edges = single_fork["edges"]
    points = single_fork["points"]

    n_edges = 2 * len(edges)
    n_points = len(points)
    n_offsets = n_edges // 2 + 1

    str_points = " ".join(map(str, points.ravel()))
    str_offsets = " ".join(map(str, range(0, n_edges + 1, 2)))
    str_connectivity = " ".join(map(str, edges.ravel()))

    str_file = f"""# vtk DataFile Version 5.1
vtk output
ASCII
DATASET POLYDATA
POINTS {n_points} float
{str_points} 
LINES {n_offsets} {n_edges}
OFFSETS vtktypeint64
{str_offsets} 
CONNECTIVITY vtktypeint64
{str_connectivity} """

    with open(path, "w") as vtkfile:
        vtkfile.write(str_file)

    print(str_file)

    node_properties, edge_properties = tested.VTK.read(path)

    expected_node_properties = pd.DataFrame(
        single_fork["points"], columns=["x", "y", "z"], dtype=np.float32
    )

    pdt.assert_frame_equal(node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        single_fork["edges"], columns=["start_node", "end_node"], dtype=np.int64
    )

    pdt.assert_frame_equal(edge_properties, expected_edge_properties)

    path = module_tmpdir.join("single_fork_vtk_2.vtk")
    tested.VTK.write(path, node_properties, edge_properties)
    node_properties, edge_properties = tested.VTK.read(path)

    pdt.assert_frame_equal(node_properties, expected_node_properties)
    pdt.assert_frame_equal(edge_properties, expected_edge_properties)


def test_vtk_read__fork_multiple_properties(single_fork_props, module_tmpdir):
    def str_data_entry(data, entry, vtk_dtype_str):
        values = data[entry]
        str_vals = " ".join(map(str, values))
        return f"{entry} 1 {len(values)} {vtk_dtype_str}\n{str_vals} "

    path = module_tmpdir.join("single_fork_props_vtk_1.vtk")

    edges = single_fork_props["edges"]
    points = single_fork_props["points"]
    point_data = single_fork_props["point_data"]
    edge_data = single_fork_props["edge_data"]

    n_edges = 2 * len(edges)
    n_points = len(points)
    n_offsets = n_edges // 2 + 1

    str_points = " ".join(map(str, points.ravel()))
    str_offsets = " ".join(map(str, range(0, n_edges + 1, 2)))
    str_connectivity = " ".join(map(str, edges.ravel()))

    n_diameters = len(point_data["diameter"])
    str_diameters = " ".join(map(str, point_data["diameter"]))

    str_file = f"""# vtk DataFile Version 5.1
vtk output
ASCII
DATASET POLYDATA

POINTS {n_points} float
{str_points}

LINES {n_offsets} {n_edges}
OFFSETS vtktypeint64
{str_offsets}

CONNECTIVITY vtktypeint64
{str_connectivity}

POINT_DATA {n_points}
FIELD FieldData {len(point_data)}
{str_data_entry(point_data, "diameter", "float")}
{str_data_entry(point_data, "point_property1", "double")}
{str_data_entry(point_data, "point_property2", "double")}

CELL_DATA {n_edges // 2}
Field FieldData 3
{str_data_entry(edge_data, "type", "char")}
{str_data_entry(edge_data, "edge_property1", "float")}
{str_data_entry(edge_data, "edge_property2", "float")}
"""
    with open(path, "w") as vtkfile:
        vtkfile.write(str_file)

    print(str_file)

    node_properties, edge_properties = tested.VTK.read(path)

    expected_node_properties = pd.DataFrame(
        {
            "x": single_fork_props["points"][:, 0],
            "y": single_fork_props["points"][:, 1],
            "z": single_fork_props["points"][:, 2],
            "diameter": single_fork_props["point_data"]["diameter"],
            "point_property1": single_fork_props["point_data"]["point_property1"],
            "point_property2": single_fork_props["point_data"]["point_property2"],
        },
        columns=["x", "y", "z", "diameter", "point_property1", "point_property2"],
    )

    pdt.assert_frame_equal(node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        {
            "start_node": single_fork_props["edges"][:, 0],
            "end_node": single_fork_props["edges"][:, 1],
            "type": single_fork_props["edge_data"]["type"],
            "edge_property1": single_fork_props["edge_data"]["edge_property1"],
            "edge_property2": single_fork_props["edge_data"]["edge_property2"],
        }
    )

    pdt.assert_frame_equal(edge_properties, expected_edge_properties)

    path = module_tmpdir.join("single_fork_props_vtk_2.vtk")
    tested.VTK.write(path, node_properties, edge_properties)

    node_properties, edge_properties = tested.VTK.read(path)
    pdt.assert_frame_equal(node_properties, expected_node_properties)
    pdt.assert_frame_equal(edge_properties, expected_edge_properties)


def test_vtk_read_write__field_data(module_tmpdir):

    path = module_tmpdir.join("field_data_vtk_1.vtk")

    str_file = f"""# vtk DataFile Version 5.1
vtk output
ASCII
DATASET POLYDATA

POINTS 3 float
0.0 0.0 0.0 1.0 1.0 1.0 2.0 2.0 2.0 

LINES 3 4
OFFSETS vtktypeint64
0 2 4

CONNECTIVITY vtktypeint64
0 1 1 2 

FIELD FieldData 2
field_property1 1 3 double
0.1 0.2 0.3 
field_property2 1 2 short
1 2 
"""

    with open(path, "w") as vtkfile:
        vtkfile.write(str_file)

    node_properties, edge_properties = tested.VTK.read(path)

    expected_node_properties = pd.DataFrame(
        {
            "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "z": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "field_property1": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        },
        columns=["x", "y", "z", "field_property1"],
    )

    pdt.assert_frame_equal(node_properties, expected_node_properties)

    expected_edge_properties = pd.DataFrame(
        {
            "start_node": np.array([0, 1], dtype=np.int64),
            "end_node": np.array([1, 2], dtype=np.int64),
            "field_property2": np.array([1, 2], dtype=np.int16),
        },
        columns=["start_node", "end_node", "field_property2"],
    )

    pdt.assert_frame_equal(edge_properties, expected_edge_properties)

    path = module_tmpdir.join("field_data_vtk_2.vtk")
    tested.VTK.write(path, node_properties, edge_properties)
    node_properties, edge_properties = tested.VTK.read(path)

    pdt.assert_frame_equal(node_properties, expected_node_properties)
    pdt.assert_frame_equal(node_properties, expected_node_properties)


def test_vtk_read__wrong_field_data(module_tmpdir):

    path = module_tmpdir.join("wrong_field_data_vtk.vtk")

    with open(path, "w") as vtkfile:
        vtkfile.write(
            "# vtk DataFile Version 5.1\n"
            "vtk output\n"
            "ASCII\n"
            "DATASET POLYDATA\n"
            "POINTS 3 float\n"
            "0.0 0.0 0.0 1.0 1.0 1.0 2.0 2.0 2.0 \n"
            "LINES 3 4\n"
            "OFFSETS vtktypeint64\n"
            "0 2 4 \n"
            "CONNECTIVITY vtktypeint64"
            "0 1 1 2 \n"
            "FIELD FieldData 1\n"
            "field_property1 1 4 double\n"
            "0.1 0.2 0.3 0.4 "
        )

    with pytest.raises(ValueError):
        tested.VTK.read(path)


def test_format_conversions():

    node_properties = pd.DataFrame(
        {
            "x": np.array([1332.1, 1331.61, 1332.14, 1332.55], dtype=np.float32),
            "y": np.array([443.8, 443.863, 443.581, 444.087], dtype=np.float32),
            "z": np.array([1911.0, 1910.78, 1910.99, 1911.09], dtype=np.float32),
            "point_property1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "point_property2": np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": np.array([0, 0, 0], dtype=np.int64),
            "end_node": np.array([1, 2, 3], dtype=np.int64),
            "edge_property1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "edge_property2": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "edge_property3": np.array([3, 2, 1], dtype=np.int8),
        }
    )

    v0 = PointGraph(node_properties, edge_properties)

    with NamedTemporaryFile(suffix=".h5") as tfile:

        filename = tfile.name

        v0.save_hdf5(filename)
        v1 = PointGraph.load_hdf5(filename)
        pdt.assert_frame_equal(v1.node_properties, v0.node_properties)
        pdt.assert_frame_equal(v1.edge_properties, v0.edge_properties)

    with NamedTemporaryFile(suffix=".vtk") as tfile:

        filename = tfile.name

        v1.save_vtk(filename)
        v2 = PointGraph.load_vtk(filename)

        pdt.assert_frame_equal(v2.node_properties, v0.node_properties)
        pdt.assert_frame_equal(v2.edge_properties, v0.edge_properties)
