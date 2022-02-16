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
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

import vascpy.conversion
from vascpy.exceptions import VasculatureAPIError
from vascpy.point_graph import features, io
from vascpy.utils.adjacency import AdjacencyMatrix

L = logging.getLogger(__name__)


@dataclass
class PointGraph:
    """Generic data structure for the representation of points/edges graphs"""

    _node_properties: pd.DataFrame
    _edge_properties: pd.DataFrame

    @classmethod
    def from_datasets(
        cls,
        points: np.ndarray,
        edges: np.ndarray,
        point_data: Optional[Dict] = None,
        edge_data: Optional[Dict] = None,
    ):
        """Create a point graph from numpy datasets of points, edges and dictionaries
        specifying the point and edge_data.
        """
        node_properties = pd.DataFrame(points, columns=["x", "y", "z"])
        edge_properties = pd.DataFrame(edges, columns=["start_node", "end_node"])

        if point_data is not None:
            for key, values in point_data.items():
                node_properties[key] = values

        if edge_data is not None:
            for key, values in edge_data.items():
                edge_properties[key] = values

        return cls(node_properties, edge_properties)

    @property
    def degrees(self):
        """Return all the node degrees"""
        return self.adjacency_matrix.degrees

    @property
    def node_properties(self):
        """Extract node_properties dataframe column as an array"""
        return self._node_properties

    @property
    def edge_properties(self):
        """Extract edge_properties dataframe column as an array"""
        return self._edge_properties

    @property
    def points(self):
        """Returns the points of the point graph"""
        return self._node_properties.loc[:, ["x", "y", "z"]].to_numpy()

    @points.setter
    def points(self, new_points):
        """points setter"""
        self._node_properties.loc[:, ["x", "y", "z"]] = new_points

    @property
    def edges(self):
        """Returns the edges of the point graph"""
        return self._edge_properties.loc[:, ["start_node", "end_node"]].to_numpy()

    @edges.setter
    def edges(self, new_edges):
        """edges setter"""
        self._edge_properties.loc[:, ["start_node", "end_node"]] = new_edges

    @property
    def segment_points(self):
        """Returns points for starts and ends of segments"""
        return self.points[self.edges.T]

    @classmethod
    def load_hdf5(cls, filepath):
        """Load and HDF5 file"""
        return cls(*io.HDF5.read(filepath))

    def save_hdf5(self, filepath):
        """Save data using the hdf5 format"""
        io.HDF5.write(filepath, self.node_properties, self.edge_properties)

    @classmethod
    def load_vtk(cls, filepath):
        """Load a VTK file"""
        return cls(*io.VTK.read(filepath))

    def save_vtk(self, filepath):
        """Save data using the VTK format"""
        io.VTK.write(filepath, self.node_properties, self.edge_properties)

    @property
    def n_nodes(self):
        """Number of nodes"""
        return len(self._node_properties)

    @property
    def n_edges(self):
        """Number of edges"""
        return len(self._edge_properties)

    @property
    def adjacency_matrix(self):
        """Returns sparse adjacency matrix of the nodes"""
        return AdjacencyMatrix(self.edges, n_vertices=self.n_nodes)

    def remove(
        self,
        node_indices: Optional[np.ndarray] = None,
        edge_indices: Optional[np.ndarray] = None,
    ):
        """
        Remove node and edge indices from the point graph.
        """
        mask_nodes_to_keep = np.ones(self.n_nodes, dtype=bool)

        if node_indices is not None:
            mask_nodes_to_keep[np.asarray(node_indices)] = False

        # we keep the edges both nodes of which are not removed
        mask_edges_to_keep = np.all(mask_nodes_to_keep[self.edges], axis=1)

        if edge_indices is not None:
            mask_edges_to_keep[np.asarray(edge_indices)] = False

        # Let the remaining edges determine the remaining nodes. Unreferenced nodes are not allowed
        # by the vasculature specs
        mask_nodes_to_keep[:] = False
        mask_nodes_to_keep[self.edges[mask_edges_to_keep]] = True

        self._node_properties = self._node_properties[mask_nodes_to_keep]
        self._edge_properties = self._edge_properties[mask_edges_to_keep]

        # for each node that is not kept there is a -1 shift to the left of the subsequent ids
        # the cumulative sum over the vertices that are not kept determines how many places to the
        # left each remaining id has to shift to make the new array
        # Before Reindexing :  - - 2 3 4 - - - 8 9
        # Cumulative sum    :  0 1 2 2 2 2 3 4 5 5
        # After Reindexing  :  - - 0 1 2 - - - 3 4
        # Subtracting the cumulative sum from the remaining edges will correcty reindex them
        self.edges -= np.cumsum(~mask_nodes_to_keep)[self.edges]


class PointVasculature(PointGraph):
    """
    Args:
        node_properties: Dataframe
            It contains at least 4 columns for:
                - x
                - y
                - z
                - diameter

        edge_properties: Dataframe
            It contains at least 5 columns for:
                - start_node
                - end_node
                - type
    """

    def __init__(self, node_properties, edge_properties):
        super().__init__(node_properties, edge_properties)
        self._validate_node_properties(self._node_properties)
        self._validate_edge_properties(self._edge_properties)

    @staticmethod
    def _validate_node_properties(node_properties):
        """Validate the minimum permitted columns in node properties"""
        for column in ("x", "y", "z", "diameter"):
            if column not in node_properties:
                raise VasculatureAPIError(f"{column} is not present in node properties")

    @staticmethod
    def _validate_edge_properties(edge_properties):
        """Validate the minimum permitted columns in edge properties"""
        for column in ("start_node", "end_node", "type"):
            if column not in edge_properties:
                raise VasculatureAPIError(f"{column} is not present in edge properties")

    @classmethod
    def load(cls, filepath: str):
        """Load point graph from file"""
        warnings.warn(
            (
                "load will be deprecated in version 1.0.0 in favor of the explicit loaders:\n"
                "load_sonata, load_hdf5 or load_vtk"
            ),
            DeprecationWarning,
        )
        filepath = str(filepath)
        if filepath.endswith(".h5"):
            return cls.load_hdf5(filepath)
        if filepath.endswith(".vtk"):
            return cls.load_vtk(filepath)
        raise VasculatureAPIError(f"{filepath} extension is unknown.")

    def save(self, filepath: str):
        """Writes the morphology to file. The extension of the file, either vtk or h5 will
        determine the format
        """
        warnings.warn(
            (
                "save will be deprecated in version 1.0.0 in favor of the explicit writers:\n"
                "save_sonata, save_hdf5 or save_vtk"
            ),
            DeprecationWarning,
        )
        filepath = str(filepath)
        if filepath.endswith(".h5"):
            self.save_hdf5(filepath)
        elif filepath.endswith(".vtk"):
            self.save_vtk(filepath)
        else:
            raise VasculatureAPIError(f"{filepath} extension is unknown.")

    @classmethod
    def load_sonata(cls, filepath):
        """Load a SONATA file"""
        return cls(*io.SONATA.read(filepath))

    def save_sonata(self, filepath):
        """Save data using the SONATA specification"""
        io.SONATA.write(filepath, self.node_properties, self.edge_properties)

    @property
    def diameters(self):
        """Returns the diameters of the point graph"""
        return self._node_properties.loc[:, "diameter"].to_numpy()

    @diameters.setter
    def diameters(self, new_diameters):
        """diameters setter"""
        self._node_properties.loc[:, "diameter"] = new_diameters

    @property
    def edge_types(self):
        """Returns the edge types of the point graph"""
        return self._edge_properties.loc[:, "type"].to_numpy()

    @edge_types.setter
    def edge_types(self, new_types):
        """edge types setter"""
        self._edge_properties.loc[:, "type"] = new_types

    @property
    def segment_diameters(self):
        """Returns diameters for starts and ends of segments"""
        return self.diameters[self.edges.T]

    @property
    def length(self):
        """Returns the total length of the vasculature"""
        return features.segment_lengths(self).sum()

    @property
    def area(self):
        """Returns the total area of the vasculature"""
        return features.segment_lateral_areas(self).sum()

    @property
    def volume(self):
        """Returns the total volume of the vasculature"""
        return features.segment_volumes(self).sum()

    def as_section_graph(self):
        """Converts the point graph to a section graph"""
        return vascpy.conversion.convert_point_to_section_vasculature(self)


def add_section_annotation(point_vasculature, annotation_name, section_ids, segment_ids):
    """If the point graph is converted from a section graph, add a positional annotation
    using the section and segment ids of the section graph.

    Args:
        point_vasculature: PointVasculature

        annotation_name: string
            The name of the annotation

        section_ids: array
            The section ids of each entry in the annotation

        segment_Ids: array
            The segment local ids of each entry within the respective section
    """
    n_entries = len(section_ids)
    if n_entries != len(segment_ids):
        raise VasculatureAPIError("Section and segment ids have different lengths.")

    edge_properties = point_vasculature.edge_properties

    # check if the point vasculature has a mapping to a section one
    if not ("section_id" in edge_properties and "segment_id" in edge_properties):
        raise VasculatureAPIError(
            "PointVasculature edge properties have not section_id, segment_id columns."
            " Construct a PointVasculature from a SectionVasculature create them."
        )

    old_index = edge_properties.index

    edge_properties.index = pd.MultiIndex.from_arrays(
        (
            edge_properties["section_id"].to_numpy(),
            edge_properties["segment_id"].to_numpy(),
        )
    )

    # initialze with -1 if the column doesn't exist
    if annotation_name not in edge_properties:
        edge_properties[annotation_name] = -1

    # multi index slicing helper
    index_slice = pd.IndexSlice[section_ids, segment_ids]
    edge_properties.loc[index_slice, annotation_name] = np.arange(n_entries, dtype=np.int64)

    edge_properties.index = old_index
