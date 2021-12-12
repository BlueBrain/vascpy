import os

from numpy import testing as npt
from pandas import testing as pdt

from vascpy.section_vasculature import SectionVasculature

_PATH = os.path.dirname(os.path.abspath(__file__))


def test_full_cycle():

    filepath = os.path.join(_PATH, "data/h5/capillary_with_loops.h5")

    s1 = SectionVasculature.load(filepath)
    p1 = s1.as_point_graph()

    s2 = p1.as_section_graph()
    p2 = s2.as_point_graph()

    npt.assert_array_equal(p1.edges, p2.edges)
    npt.assert_allclose(p1.points, p2.points)
    npt.assert_allclose(p1.diameters, p2.diameters)


def test_full_cycle__wout_section_ids():

    filepath = os.path.join(_PATH, "data/h5/capillary_with_loops.h5")

    s1 = SectionVasculature.load(filepath)
    p1 = s1.as_point_graph()
    p1.edge_properties.drop(columns=["section_id", "segment_id"], inplace=True)

    s2 = p1.as_section_graph()
    p2 = s2.as_point_graph()

    npt.assert_array_equal(p1.edges, p2.edges)
    npt.assert_allclose(p1.points, p2.points)
    npt.assert_allclose(p1.diameters, p2.diameters)
