import pathlib

import numpy as np
import pytest
from numpy import testing as npt

from vascpy.exceptions import VasculatureAPIError
from vascpy.point_vasculature import PointVasculature, add_section_annotation
from vascpy.section_vasculature import SectionVasculature

_DATAPATH = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def point_vasculature_with_mapping():
    path = _DATAPATH / "h5/fork_diameters_section_spec.h5"
    return SectionVasculature.load(path).as_point_graph()


@pytest.fixture
def point_vasculature_wout_mapping():
    path = _DATAPATH / "h5/fork_diameters_section_spec.h5"
    p_graph = SectionVasculature.load(path).as_point_graph()
    p_graph.edge_properties.drop(columns=["section_id", "segment_id"], inplace=True)
    return p_graph


def test_section_annotation(point_vasculature_with_mapping, point_vasculature_wout_mapping):

    section_ids = np.array([0, 2], dtype=np.uint32)
    segment_ids = np.array([0, 0], dtype=np.uint32)

    annotation_name = "foo"

    # no multi index
    with pytest.raises(VasculatureAPIError):
        add_section_annotation(
            point_vasculature_wout_mapping, annotation_name, section_ids, segment_ids
        )

    # diff length of section segment ids
    with pytest.raises(VasculatureAPIError):
        add_section_annotation(
            point_vasculature_with_mapping, annotation_name, section_ids[:-1], segment_ids
        )

    add_section_annotation(
        point_vasculature_with_mapping, annotation_name, section_ids, segment_ids
    )

    values = point_vasculature_with_mapping.edge_properties.loc[:, annotation_name]
    npt.assert_array_equal(values, [0, -1, 1])

    # write on top of the existing annotation

    section_ids = np.array([1, 2], dtype=np.uint32)
    segment_ids = np.array([0, 0], dtype=np.uint32)

    add_section_annotation(
        point_vasculature_with_mapping, annotation_name, section_ids, segment_ids
    )

    values = point_vasculature_with_mapping.edge_properties.loc[:, annotation_name]
    npt.assert_array_equal(values, [0, 0, 1])
