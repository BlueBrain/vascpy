import pandas as pd
import pytest
from numpy import testing as npt

from vascpy.point_graph import features as _feat
from vascpy.point_vasculature import PointVasculature


@pytest.fixture
def node_properties():
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "z": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "diameter": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "property1": [3.0, 1.0, 4.0, 1.0, 4.0, 5.0, 6.0, 8.0, 9.0, 0.0],
        }
    )


@pytest.fixture
def edge_properties():
    return pd.DataFrame(
        {
            "start_node": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "end_node": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "type": [1, 0, 1, 0, 1, 0, 1, 0, 1],
            "property1": [9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )


@pytest.fixture
def point_vasculature(node_properties, edge_properties):
    return PointVasculature(node_properties, edge_properties)


def test_segment_volumes(point_vasculature):
    segment_volumes = _feat.segment_volumes(point_vasculature)
    npt.assert_allclose(
        segment_volumes,
        [
            16.77764414,
            27.66044034,
            41.26393559,
            57.5881299,
            76.63302325,
            98.39861565,
            122.8849071,
            150.0918976,
            180.0195872,
        ],
    )


def test_segment_surface_areas(point_vasculature):
    segment_areas = _feat.segment_lateral_areas(point_vasculature)
    npt.assert_allclose(
        segment_areas,
        [
            19.82255347,
            25.48614018,
            31.14972689,
            36.8133136,
            42.4769003,
            48.14048701,
            53.80407372,
            59.46766042,
            65.13124713,
        ],
    )


def test_segment_lengths(point_vasculature):
    segment_lengths = _feat.segment_lengths(point_vasculature)
    npt.assert_allclose(segment_lengths, [1.73205081] * 9)
