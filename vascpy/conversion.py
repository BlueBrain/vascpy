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
import tempfile
from collections import deque

import h5py
import numpy as np
import pandas as pd

import vascpy.point_vasculature
import vascpy.section_vasculature
from vascpy import specs
from vascpy.utils.geometry import unique_points

# from vascpy.section_graph.io import save_section_graph_data
from vascpy.utils.section_creation import create_chains, reconstruct_chains_using_groups

L = logging.getLogger(__name__)


class ColsPoints:
    """Enumeration for point array columns"""

    # pylint: disable=too-few-public-methods

    X = 0
    Y = 1
    Z = 2


class ColsEdges:
    """Enumeration for edge property columns"""

    # pylint: disable=too-few-public-methods

    BEG_NODE = 0
    END_NODE = 1
    EDGE_TYPE = 2
    SECTION_ID = 3
    SEGMENT_ID = 4


def convert_section_to_point_vasculature(section_vasculature):
    """Convert section graph to point graph data"""
    points, diameters, edge_data = _sections_to_point_connectivity(section_vasculature.sections)

    SPEC = specs.SpecPointVasculature

    node_properties = pd.DataFrame(
        {
            SPEC.X.name: points[:, 0].astype(SPEC.X.dtype),
            SPEC.Y.name: points[:, 1].astype(SPEC.Y.dtype),
            SPEC.Z.name: points[:, 2].astype(SPEC.Z.dtype),
            SPEC.D.name: diameters.astype(SPEC.D.dtype),
        },
        index=pd.RangeIndex(start=0, stop=len(points), dtype=np.int64),
        columns=[t.name for t in SPEC.NODE_PROPERTIES],
    )

    edge_properties = pd.DataFrame(
        {
            SPEC.SOURCE.name: edge_data[:, 0].astype(SPEC.SOURCE.dtype),
            SPEC.TARGET.name: edge_data[:, 1].astype(SPEC.TARGET.dtype),
            SPEC.TYPE.name: edge_data[:, 2].astype(SPEC.TYPE.dtype),
            SPEC.SECTION_ID.name: edge_data[:, 3].astype(SPEC.SECTION_ID.dtype),
            SPEC.SEGMENT_ID.name: edge_data[:, 4].astype(SPEC.SEGMENT_ID.dtype),
        },
        index=pd.RangeIndex(start=0, stop=len(edge_data), dtype=np.int64),
        columns=[t.name for t in SPEC.EDGE_PROPERTIES],
    )

    return vascpy.point_vasculature.PointVasculature(node_properties, edge_properties)


def convert_point_to_section_vasculature(point_vasculature):  # pylint: disable=R0914,R0915
    """Convert to point graph representation to section graph representation"""

    def _section_point_indices(edge_list):

        point_indices = deque()

        for (beg, end) in edge_list:
            if point_indices:
                assert beg == point_indices[-1]
                point_indices.append(end)
            else:
                point_indices.extend((beg, end))
        return np.asarray(point_indices, np.int64)

    SPEC = specs.SpecSectionHDF5

    edge_properties = point_vasculature.edge_properties

    edges = point_vasculature.edges
    points = point_vasculature.points
    diameters = point_vasculature.diameters

    edge_types = edge_properties.loc[:, "type"].to_numpy()

    if "section_id" in edge_properties:

        section_ids = edge_properties.loc[:, "section_id"].to_numpy()
        chains, chain_connectivity, edge_ids_per_chain = reconstruct_chains_using_groups(
            edges, section_ids, return_index=True
        )

    else:

        L.info("Section ids for edges were not provided. A specific ordering is not guaranteed.")
        chains, chain_connectivity, edge_ids_per_chain = create_chains(
            edges, len(points), return_index=True
        )

    section_types = np.empty(len(chains), dtype=edge_types.dtype)
    for section_index, edge_ids in enumerate(edge_ids_per_chain):

        e_types = edge_types[edge_ids]
        section_type = e_types[0]

        assert np.all(section_type == e_types), "Edge types in the same section are different"
        section_types[section_index] = section_type

    point_data = np.column_stack((points, diameters))

    # hack to create a morphio vasculature
    with tempfile.NamedTemporaryFile(suffix=".h5") as temp:

        filename = temp.name

        n_sections = len(chains)

        section_point_idx = []

        point_offset = 0

        structure = np.empty((n_sections, 2), dtype=SPEC.STRUCTURE)

        structure[:, 1] = section_types

        # crate structure
        for section_index, edge_list in enumerate(chains):

            structure[section_index, 0] = point_offset

            current_point_idx = _section_point_indices(edge_list)
            section_point_idx.append(current_point_idx)

            point_offset += len(current_point_idx)

        point_indices = np.hstack(section_point_idx)
        new_point_data = point_data[point_indices]

        with h5py.File(filename, "w") as h5f:
            h5f.create_dataset("points", data=new_point_data, dtype=SPEC.POINTS)
            h5f.create_dataset("structure", data=structure, dtype=SPEC.STRUCTURE)
            h5f.create_dataset("connectivity", data=chain_connectivity, dtype=SPEC.CONNECTIVITY)

        section_vasculature = vascpy.section_vasculature.SectionVasculature.load(filename)

    return section_vasculature


def _sections_to_point_connectivity(sections):
    """Given a list of section with duplicate points, returns
    the point connectivity.

    1. Converts the sections to edges by considering the duplicate points
       as separate vertices initially.

    2. The unique points are found without changing the ordering and the duplicate
       points collapse to the same vertices. Thus, the edges refering to duplicate
       points will become incident to the collapsed ones.

    Args:
        sections: list of morphio sections

    Note:
        This function relies on the morphio representation where there
        are duplicate points for the forking points. It will not work
        without duplicate points

    Returns:
        unique_points:
            All the unique points in the sections maintaining the initial ordering
        unique_diameters:
            All the unique diameters in the sections maintaining the initial ordering
        remapped_edges:
            The initial edges remapped to reflect the collapsed duplicate points
    """
    # pylint: disable=too-many-locals
    points, diameters, edge_properties = [], [], []
    edge_offsets = np.zeros(len(sections) + 1, dtype=np.int64)

    total_points = 0
    for i, section in enumerate(sections):

        section_points = section.points
        n_points = len(section_points)
        n_edges = n_points - 1

        edge_props = np.empty((n_edges, 5), dtype=np.int64)

        indices = np.arange(n_points, dtype=np.int32)
        edge_props[:, 0] = total_points + indices[:-1]
        edge_props[:, 1] = total_points + indices[1:]
        edge_props[:, 2] = int(section.type)
        edge_props[:, 3] = section.id
        edge_props[:, 4] = indices[:-1]

        points.append(section_points)
        diameters.append(section.diameters)
        edge_properties.append(edge_props)

        # edge_counts_per_section[i]
        total_points += n_points
        edge_offsets[i + 1] = edge_offsets[i] + n_points - 1

    points = np.vstack(points)
    diameters = np.hstack(diameters)
    edge_properties = np.vstack(edge_properties)

    unique_indices, inverse_mapping = unique_points(points, decimals=3)

    cols_edges = [ColsEdges.BEG_NODE, ColsEdges.END_NODE]
    # inverse mappings map the edge vertices to the unique ones
    edge_properties[:, cols_edges] = inverse_mapping[edge_properties[:, cols_edges]]

    return points[unique_indices], diameters[unique_indices], edge_properties
