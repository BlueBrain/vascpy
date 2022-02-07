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
# pylint: disable=invalid-name, import-outside-toplevel, too-many-locals
import logging
import typing
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd

from vascpy import specs

L = logging.getLogger(__name__)


class SONATA:
    """HDF5 SONATA reader and writer"""

    @staticmethod
    def read(filepath: str):
        """Read an hdf5 file created using the SONATA specification"""
        import libsonata

        SPEC = specs.SpecSONATA

        pop = libsonata.NodeStorage(filepath).open_population("vasculature")
        property_names = pop.attribute_names

        all_ids = pop.select_all()

        # support for legacy attribute names
        if "start_node_id" in property_names and "end_node_id" in property_names:

            beg_ids = pop.get_attribute("start_node_id", all_ids)
            end_ids = pop.get_attribute("end_node_id", all_ids)

        else:

            beg_ids = pop.get_attribute("start_node", all_ids)
            end_ids = pop.get_attribute("end_node", all_ids)

        # edges integer types cannot be uint64 because of numpy's absurd promotion rules
        # https://github.com/numpy/numpy/issues/5745
        edge_properties_dict = {
            "start_node": beg_ids.astype(np.int64),
            "end_node": end_ids.astype(np.int64),
        }

        for name, dtype in SPEC.SONATA_EDGE_DTYPES.items():
            if name not in edge_properties_dict:
                edge_properties_dict[name] = pop.get_attribute(name, all_ids).astype(dtype)

        uids = np.unique(np.hstack([beg_ids, end_ids]))

        # ensure consecutive ids
        assert np.all(uids == np.arange(uids.size, dtype=uids.dtype))

        node_properties_dict: typing.Dict[str, np.ndarray] = {
            name: np.empty(uids.size, dtype=dtype)
            for name, dtype in SPEC.SONATA_POINT_DTYPES.items()
        }

        for name, dtype in SPEC.SONATA_POINT_DTYPES.items():
            node_properties_dict[name][beg_ids] = pop.get_attribute(f"start_{name}", all_ids)
            node_properties_dict[name][end_ids] = pop.get_attribute(f"end_{name}", all_ids)

        return (
            pd.DataFrame(node_properties_dict, index=pd.RangeIndex(0, uids.size, dtype=np.int64)),
            pd.DataFrame(edge_properties_dict, index=pd.RangeIndex(0, pop.size, dtype=np.int64)),
        )

    @staticmethod
    def write(filepath: str, node_properties: pd.DataFrame, edge_properties: pd.DataFrame):
        """Write an hdf5 file using the SONATA specification"""
        import h5py

        SPEC = specs.SpecSONATA

        n_edges = len(edge_properties)
        beg_ids = edge_properties.loc[:, "start_node"]
        end_ids = edge_properties.loc[:, "end_node"]

        props = {
            name: edge_properties[name].astype(dtype)
            for name, dtype in SPEC.SONATA_EDGE_DTYPES.items()
        }

        for name, dtype in SPEC.SONATA_POINT_DTYPES.items():
            column_location = node_properties.columns.get_loc(name)
            props[f"start_{name}"] = node_properties.iloc[beg_ids, column_location].astype(dtype)
            props[f"end_{name}"] = node_properties.iloc[end_ids, column_location].astype(dtype)

        with h5py.File(filepath, "w") as h5f:

            population = h5f.create_group("/nodes/vasculature", track_order=True)
            population.create_dataset("node_type_id", data=np.full(n_edges, -1), dtype=np.int64)

            group = population.create_group("0")
            group.create_dataset(
                "@library/model_type",
                data=["vasculature"],
                dtype=h5py.special_dtype(vlen=str),
            )
            group.create_dataset("model_type", data=np.zeros(n_edges, dtype=np.int8))

            for name, values in props.items():
                group.create_dataset(name, data=values)


class HDF5:
    """HDF5 reader and writer"""

    @staticmethod
    def read(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load hdf5"""
        import h5py

        node_dict, edge_dict = OrderedDict(), OrderedDict()

        with h5py.File(filepath, "r") as fd:

            # pylint: disable=no-member
            node_dict["x"], node_dict["y"], node_dict["z"] = fd["points"][:].T

            for name in fd["point_properties"]:
                node_dict[name] = fd["point_properties"][name][:]

            # pylint: disable=no-member
            edge_dict["start_node"], edge_dict["end_node"] = fd["edges"][:].T

            for name in fd["edge_properties"]:
                edge_dict[name] = fd["edge_properties"][name][:]

        return (
            pd.DataFrame(node_dict, columns=node_dict.keys()),
            pd.DataFrame(edge_dict, columns=edge_dict.keys()),
        )

    @staticmethod
    def write(filepath: str, node_properties: pd.DataFrame, edge_properties: pd.DataFrame):
        """Write point graph data to file"""
        import h5py

        with h5py.File(filepath, "w") as fd:

            fd.create_dataset(
                name="points", data=node_properties.loc[:, ["x", "y", "z"]].to_numpy()
            )
            fd.create_dataset(
                name="edges", data=edge_properties.loc[:, ["start_node", "end_node"]].to_numpy()
            )

            point_properties_group = fd.create_group("point_properties", track_order=True)
            node_properties = node_properties.drop(columns=["x", "y", "z"])
            for name, values in node_properties.items():
                point_properties_group.create_dataset(name, data=values.to_numpy())

            print(edge_properties)
            edge_properties_group = fd.create_group("edge_properties", track_order=True)
            edge_properties = edge_properties.drop(columns=["start_node", "end_node"])
            for name, values in edge_properties.items():
                edge_properties_group.create_dataset(name, data=values.to_numpy())


class VTK:
    """VTK reader and writer"""

    @staticmethod
    def read(filepath):
        """Extracts from a vtk file the points, edges, radii and types"""
        import vtk
        from vtk.util import numpy_support as ns

        def get_points(polydata):
            vpoints = polydata.GetPoints()
            return ns.vtk_to_numpy(vpoints.GetData())

        def get_edges(polydata):
            vlines = polydata.GetLines()
            nmp_lines = ns.vtk_to_numpy(vlines.GetData())
            n_rows = int(len(nmp_lines) / 3)
            return nmp_lines.reshape(n_rows, 3)[:, (1, 2)].astype(np.int64)

        def iter_attributes(data):
            n_arrays = data.GetNumberOfArrays()
            return (
                (data.GetArrayName(i), ns.vtk_to_numpy(data.GetArray(i))) for i in range(n_arrays)
            )

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(filepath))
        reader.Update()

        polydata = reader.GetOutput()
        node_dict, edge_dict = OrderedDict(), OrderedDict()

        node_dict["x"], node_dict["y"], node_dict["z"] = get_points(polydata).T
        edge_dict["start_node"], edge_dict["end_node"] = get_edges(polydata).T

        node_dict.update(iter_attributes(polydata.GetPointData()))
        edge_dict.update(iter_attributes(polydata.GetCellData()))

        n_nodes = len(node_dict["x"])
        n_edges = len(edge_dict["start_node"])

        for name, data in iter_attributes(polydata.GetFieldData()):

            if len(data) == n_nodes:
                node_dict[name] = data
            elif len(data) == n_edges:
                edge_dict[name] = data
            else:
                raise ValueError(f"Incompatible {name} property size with both nodes and edges")

        return (
            pd.DataFrame(node_dict, columns=node_dict.keys(), index=pd.RangeIndex(n_nodes)),
            pd.DataFrame(edge_dict, columns=edge_dict.keys(), index=pd.RangeIndex(n_edges)),
        )

    @staticmethod
    def write(filepath, node_properties, edge_properties):
        """Write to vtk file"""
        import vtk
        from vtk.util import numpy_support as ns

        def vtk_points(points):
            """Converts an array of numpy points to vtk points"""
            vpoints = vtk.vtkPoints()
            vpoints.SetData(ns.numpy_to_vtk(points.copy(), deep=1))
            return vpoints

        def vtk_lines(edges):
            """Converts a list of edges into vtk lines"""
            vlines = vtk.vtkCellArray()

            n_edges = edges.shape[0]

            arr = np.empty((n_edges, 3), order="C", dtype=np.int64)

            arr[:, 0] = 2
            arr[:, 1:] = edges

            # crucial to deep copy the data!!!
            vlines.SetCells(edges.shape[0], ns.numpy_to_vtkIdTypeArray(arr, deep=1))
            return vlines

        def vtk_attribute_array(name, arr):
            """Creates a cell array with specified name and assignes the
            numpy array arr
            """
            val_arr = vtk.util.numpy_support.numpy_to_vtk(arr)
            val_arr.SetName(name)
            return val_arr

        polydata = vtk.vtkPolyData()

        polydata.SetPoints(vtk_points(node_properties.loc[:, ["x", "y", "z"]]))
        polydata.SetLines(vtk_lines(edge_properties.loc[:, ["start_node", "end_node"]]))

        node_properties = node_properties.drop(columns=["x", "y", "z"])
        for name, values in node_properties.items():
            vtk_attribute = vtk_attribute_array(name, np.ascontiguousarray(values))
            polydata.GetPointData().AddArray(vtk_attribute)

        edge_properties = edge_properties.drop(columns=["start_node", "end_node"])
        for name, values in edge_properties.items():
            vtk_attribute = vtk_attribute_array(name, np.ascontiguousarray(values))
            polydata.GetCellData().AddArray(vtk_attribute)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(filepath))
        writer.SetFileTypeToASCII()
        writer.SetInputData(polydata)
        writer.Write()
