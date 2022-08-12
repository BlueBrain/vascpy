import tempfile
from pathlib import Path

import numpy.testing as npt
import pytest

from vascpy import section_vasculature as tested

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def section_vasculature():
    return tested.SectionVasculature(DATA_DIR / "h5/capillary_with_loops.h5")


def test_loaders():

    v1 = tested.SectionVasculature(DATA_DIR / "h5/capillary_with_loops.h5")
    v2 = tested.SectionVasculature.load(DATA_DIR / "h5/capillary_with_loops.h5")
    v3 = tested.SectionVasculature.load_hdf5(DATA_DIR / "h5/capillary_with_loops.h5")

    npt.assert_allclose(v1.points, v2.points)
    npt.assert_allclose(v2.points, v3.points)


def test_writers(section_vasculature):

    with tempfile.NamedTemporaryFile(suffix=".h5") as tfile:

        filepath = tfile.name

        section_vasculature.save(filepath)
        v1 = tested.SectionVasculature(filepath)

        section_vasculature.save_hdf5(filepath)
        v2 = tested.SectionVasculature(filepath)

        npt.assert_allclose(v1.points, v2.points)


def test_methods(section_vasculature):
    section_vasculature.points
    section_vasculature.diameters
    section_vasculature.sections
    section_vasculature.iter()
