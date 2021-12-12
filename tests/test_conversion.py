import json
import os
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from morphio.vasculature import Vasculature
from numpy import testing as npt
from pandas import testing as pdt

from vascpy import SectionVasculature
from vascpy import conversion as tested

np.set_printoptions(threshold=sys.maxsize)
from vascpy.exceptions import VasculatureAPIError
from vascpy.point_vasculature import PointGraph

DATAPATH = Path(__file__).parent / "data"


class MockSection:
    __slots__ = "id", "type", "points", "diameters", "successors", "predecessors"

    def __init__(self, sid, points, diameters):

        self.id = sid
        self.type = 1
        self.points = points
        self.diameters = diameters
        self.successors = []
        self.predecessors = []

    def __str__(self):
        return "< {} ps: {} ss: {} >".format(
            self.id, [s.id for s in self.predecessors], [s.id for s in self.successors]
        )

    __repr__ = __str__


def test_sections_to_point_connectivity__one_section():

    points = np.random.random((5, 3))
    diameters = np.random.random(5)

    section = MockSection(0, points, diameters)

    r_points, r_diameters, r_edge_properties = tested._sections_to_point_connectivity([section])

    npt.assert_allclose(points, r_points)
    npt.assert_allclose(diameters, r_diameters)

    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.BEG_NODE], [0, 1, 2, 3])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.END_NODE], [1, 2, 3, 4])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SECTION_ID], [0, 0, 0, 0])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SEGMENT_ID], [0, 1, 2, 3])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.EDGE_TYPE], [1, 1, 1, 1])


def test_sections_to_point_connectivity__bifurcation():

    points1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    diameters1 = np.array([0.0, 1.0, 2.0])

    points2 = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    diameters2 = np.array([2.0, 3.0, 4.0])

    points3 = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    diameters3 = np.array([2.0, 5.0, 6.0])

    s1 = MockSection(0, points1, diameters1)
    s2 = MockSection(1, points2, diameters2)
    s3 = MockSection(2, points3, diameters3)

    s1.successors = [s2, s3]
    s2.predecessors = [s1]
    s3.predecessors = [s1]

    sections = [s1, s2, s3]

    r_points, r_diameters, r_edge_properties = tested._sections_to_point_connectivity(sections)

    npt.assert_allclose(
        r_points,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ],
    )

    npt.assert_allclose(r_diameters, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.BEG_NODE], [0, 1, 2, 3, 2, 5])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.END_NODE], [1, 2, 3, 4, 5, 6])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SECTION_ID], [0, 0, 1, 1, 2, 2])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SEGMENT_ID], [0, 1, 0, 1, 0, 1])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.EDGE_TYPE], [1, 1, 1, 1, 1, 1])


def test_sections_to_point_connectivity__bifurcation_2():

    points1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    diameters1 = np.array([0.0, 1.0, 2.0])

    points2 = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    diameters2 = np.array([2.0, 3.0, 4.0])

    points3 = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    diameters3 = np.array([2.0, 5.0, 6.0])

    s1 = MockSection(0, points1, diameters1)
    s2 = MockSection(1, points2, diameters2)
    s3 = MockSection(2, points3, diameters3)

    s1.successors = [s2, s3]
    s2.predecessors = [s1]
    s3.predecessors = [s1]

    sections = [s1, s2, s3]

    r_points, r_diameters, r_edge_properties = tested._sections_to_point_connectivity(sections)

    npt.assert_allclose(
        r_points,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ],
    )

    npt.assert_allclose(r_diameters, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.BEG_NODE], [0, 1, 2, 3, 2, 5])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.END_NODE], [1, 2, 3, 4, 5, 6])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SECTION_ID], [0, 0, 1, 1, 2, 2])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SEGMENT_ID], [0, 1, 0, 1, 0, 1])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.EDGE_TYPE], [1, 1, 1, 1, 1, 1])


def test_sections_to_point_connectivity__one_loop():
    """
                 s2
              |-------|
        s1    |       |    s4
    -----------       ------------
              |       |
              |-------|
                 s3
    """

    points1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    diameters1 = np.array([0.0, 1.0, 2.0])

    points2 = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])

    diameters2 = np.array([2.0, 3.0, 4.0])

    points3 = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [4.0, 4.0, 4.0]])

    diameters3 = np.array([2.0, 5.0, 4.0])

    points4 = np.array([[4.0, 4.0, 4.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    diameters4 = np.array([4.0, 6.0, 7.0])

    s1 = MockSection(0, points1, diameters1)
    s2 = MockSection(1, points2, diameters2)
    s3 = MockSection(2, points3, diameters3)
    s4 = MockSection(3, points4, diameters4)

    s1.successors = [s2, s3]

    s2.predecessors = [s1]
    s2.successors = [s3, s4]

    s3.predecessors = [s1]
    s3.successors = [s2, s4]

    s4.predecessors = [s2, s3]

    sections = [s1, s2, s3, s4]
    r_points, r_diameters, r_edge_properties = tested._sections_to_point_connectivity(sections)

    npt.assert_allclose(
        r_points,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
        ],
        verbose=True,
    )

    npt.assert_allclose(r_diameters, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], verbose=True)

    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.BEG_NODE], [0, 1, 2, 3, 2, 5, 4, 6])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.END_NODE], [1, 2, 3, 4, 5, 4, 6, 7])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SECTION_ID], [0, 0, 1, 1, 2, 2, 3, 3])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.SEGMENT_ID], [0, 1, 0, 1, 0, 1, 0, 1])
    npt.assert_equal(r_edge_properties[:, tested.ColsEdges.EDGE_TYPE], [1, 1, 1, 1, 1, 1, 1, 1])


def test_sections_to_point_connectivity__capillary_data():

    m = Vasculature(DATAPATH / "h5/capillary_with_loops.h5")

    points, diameters, edge_properties = tested._sections_to_point_connectivity(m.sections)

    with open(DATAPATH / "capillary_with_loops_point_connectivity.json", "r") as fd:
        expected_data = json.load(fd)

    npt.assert_allclose(points, expected_data["points"])
    npt.assert_allclose(diameters, expected_data["diameters"])

    expected_props = expected_data["edge_properties"]

    npt.assert_allclose(edge_properties[:, tested.ColsEdges.BEG_NODE], expected_props["beg_node"])
    npt.assert_allclose(edge_properties[:, tested.ColsEdges.END_NODE], expected_props["end_node"])
    npt.assert_allclose(
        edge_properties[:, tested.ColsEdges.SECTION_ID], expected_props["section_id"]
    )
    npt.assert_allclose(
        edge_properties[:, tested.ColsEdges.SEGMENT_ID], expected_props["segment_id"]
    )
    npt.assert_allclose(edge_properties[:, tested.ColsEdges.EDGE_TYPE], expected_props["edge_type"])


def _assert_point_graph_files_equal(filepath1, filepath2):

    if filepath1.endswith(".h5"):
        pgraph1 = PointGraph.load_hdf5(filepath1)
    else:
        pgraph1 = PointGraph.load_vtk(filepath1)

    if filepath2.endswith(".h5"):
        pgraph2 = PointGraph.load_hdf5(filepath2)
    else:
        pgraph2 = PointGraph.load_vtk(filepath2)

    print(pgraph1.node_properties)
    print(pgraph2.node_properties)

    print(pgraph1.edge_properties)
    print(pgraph2.edge_properties)

    pdt.assert_frame_equal(pgraph1.node_properties, pgraph2.node_properties)
    pdt.assert_frame_equal(pgraph1.edge_properties, pgraph2.edge_properties)


def _assert_section_graph_files_equal(filepath1, filepath2):

    morph1 = SectionVasculature.load(filepath1)
    morph2 = SectionVasculature.load(filepath2)

    for section1, section2 in zip(morph1.sections, morph2.sections):

        assert section1.id == section2.id
        npt.assert_allclose(section1.points, section2.points)
        npt.assert_allclose(section1.diameters, section2.diameters)


"""
def test_convert__point_to_point():

    vtk_input_filepath = str(DATAPATH / 'vtk/fork_edge_radii_edges_field_data.vtk')
    validation_path = str(DATAPATH / 'vtk/fork_diameters_point_spec.vtk')

    with tempfile.TemporaryDirectory() as temp_dir:

        vtk_filepath = str(Path(temp_dir) / 'out.vtk')

        # should raise because there are no diameters on points
        with pytest.raises(VasculatureAPIError):
            tested.point_to_point(vtk_input_filepath, vtk_filepath, standardize=False)

        # standardization takes care of converting to the correct spec
        tested.point_to_point(vtk_input_filepath, vtk_filepath, standardize=True)
        _assert_point_graph_files_equal(validation_path, vtk_filepath)

        # curation should not change it
        tested.point_to_point(vtk_input_filepath, vtk_filepath, standardize=True, curate=True)
        _assert_point_graph_files_equal(validation_path, vtk_filepath)


        h5_filepath = str(Path(temp_dir) / 'out.h5')

        # should raise because there are no diameters on points
        with pytest.raises(VasculatureAPIError):
            tested.point_to_point(vtk_input_filepath, h5_filepath, standardize=False)

        tested.point_to_point(vtk_input_filepath, h5_filepath, standardize=True)
        _assert_point_graph_files_equal(validation_path, h5_filepath)


def test_convert_point_to_section():

    vtk_input_filepath = str(DATAPATH / 'vtk/fork_edge_radii_edges_field_data.vtk')
    validation_path = str(DATAPATH / 'h5/fork_diameters_section_spec.h5')

    with tempfile.TemporaryDirectory() as temp_dir:

        h5_filepath = str(Path(temp_dir) / 'out.h5')

        # should raise because there are no diameters on points
        with pytest.raises(VasculatureAPIError):
            tested.point_to_section(vtk_input_filepath, h5_filepath, standardize=False)

        # standardization takes care of converting to the correct spec
        tested.point_to_section(vtk_input_filepath, h5_filepath, standardize=True)
        _assert_section_graph_files_equal(validation_path, h5_filepath)

        tested.point_to_section(vtk_input_filepath, h5_filepath, standardize=True, curate=True)
        _assert_section_graph_files_equal(validation_path, h5_filepath)


def test_convert_section_to_point():

    h5_input_filepath = str(DATAPATH / 'h5/fork_diameters_section_spec.h5')
    validation_path_h5 = str(DATAPATH / 'h5/fork_diameters_point_spec.h5')
    validation_path_vtk = str(DATAPATH / 'vtk/fork_diameters_point_spec.vtk')

    with tempfile.TemporaryDirectory() as temp_dir:


        p_graph = SectionVasculature.load(h5_input_filepath).as_point_graph()
        p_graph.edge_properties.drop(columns=['section_id', 'segment_id'], inplace=True)

        h5_filepath = str(Path(temp_dir) / 'out.h5')
        p_graph.save_hdf5(h5_filepath)
        _assert_point_graph_files_equal(h5_filepath, validation_path_h5)

        vtk_filepath = str(Path(temp_dir) / 'out.vtk')
        p_graph.save_vtk(vtk_filepath)
        _assert_point_graph_files_equal(vtk_filepath, validation_path_vtk)
"""

"""
def _create_section_attribute(section_ids, segment_ids, data_dict):

    index = pd.MultiIndex.from_arrays([section_ids, segment_ids], names=['section_id', 'segment_id'])
    return pd.DataFrame(data_dict, index=index).sort_index()


def test_sections_to_point_connectivity__edge_attribute_1():

    points = np.random.random((5, 3))
    diameters = np.random.random(5)

    section = MockSection(0, points, diameters)

    section_attributes = {
        'attribute_1': _create_section_attribute([0, 0, 0], [0, 1, 2], {'col1': [0, 1, 2]}),
        'attribute_2': _create_section_attribute([0, 0, 0], [2, 3, 4], {'col1': [1, 2, 3], 'col2': [3, 4, 5]}),
        'attribute_3': _create_section_attribute([0, 0, 0], [0, 2, 5], {'col1': [6, 7, 8], 'col2': [9, 10, 11], 'col3': [12, 13, 14]})
    }

    _, _, r_edge_properties, r_edge_attributes = sections_to_point_connectivity([section], section_attributes)

    e_attribute_1 = pd.DataFrame({'col1': [0 , 1, 2]}, index=pd.Index([0, 1, 2], name='edge_id'))
    pdt.assert_frame_equal(e_attribute_1, r_edge_attributes['attribute_1'])

    e_attribute_2 = pd.DataFrame({'col1': [1 , 2, 3], 'col2': [3, 4, 5]}, index=pd.Index([2, 3, 4], name='edge_id'))
    pdt.assert_frame_equal(e_attribute_2, r_edge_attributes['attribute_2'])

    e_attribute_3 = pd.DataFrame({'col1': [6 , 7, 8], 'col2': [9, 10, 11], 'col3': [12, 13, 14]}, index=pd.Index([0, 2, 5], name='edge_id'))
    pdt.assert_frame_equal(e_attribute_3, r_edge_attributes['attribute_3'])


def test_sections_to_point_connectivity__edge_attribute_2():

    points1 = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    diameters1 = np.array([0., 1., 2.])

    points2 = np.array([[2., 2., 2.], [3., 3., 3.], [4., 4., 4.]])
    diameters2 = np.array([2., 3., 4.])

    points3 = np.array([[2., 2., 2.], [5., 5., 5.], [6., 6., 6.]])
    diameters3 = np.array([2., 5., 6.])

    s1 = MockSection(0, points1, diameters1)
    s2 = MockSection(1, points2, diameters2)
    s3 = MockSection(2, points3, diameters3)

    s1.successors = [s2, s3]
    s2.predecessors = [s1]
    s3.predecessors = [s1]

    sections = [s1, s2, s3]

    section_attributes = {
        'attribute_1': _create_section_attribute([0], [1], {'spam': [3.14]}),
        'attribute_2': _create_section_attribute([0, 1], [1, 1], {'egg': [2, 2], 'ham': ['2', '2']}),
        'attribute_3': _create_section_attribute([0, 1, 2], [1, 0, 1], {'foo': [6, 7, 8], 'lol': [9, 10, 11], 'pen': [12, 13, 14]})
    }

    _, _, _, r_edge_attributes = sections_to_point_connectivity(sections, section_attributes)

    e_attribute_1 = pd.DataFrame({'spam': [3.14]}, index=pd.Index([1], name='edge_id'))
    pdt.assert_frame_equal(e_attribute_1, r_edge_attributes['attribute_1'])

    e_attribute_2 = pd.DataFrame({'egg': [2, 2], 'ham': ['2', '2']}, index=pd.Index([1, 3], name='edge_id'))
    pdt.assert_frame_equal(e_attribute_2, r_edge_attributes['attribute_2'])

    e_attribute_3 = pd.DataFrame({'foo': [6, 7, 8], 'lol': [9, 10, 11], 'pen': [12, 13, 14]}, index=pd.Index([1, 2, 5], name='edge_id'))
    pdt.assert_frame_equal(e_attribute_3, r_edge_attributes['attribute_3'])
"""
