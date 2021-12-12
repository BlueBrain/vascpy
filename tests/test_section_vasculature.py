from pathlib import Path

import pytest

from vascpy import section_vasculature as tested

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def section_vasculature():
    return tested.SectionVasculature(DATA_DIR / "h5/capillary_with_loops.h5")


def test_methods(section_vasculature):
    section_vasculature.points
    section_vasculature.diameters
    section_vasculature.sections
    section_vasculature.iter()
