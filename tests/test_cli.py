import tempfile
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner
from numpy import testing as npt
from pandas import testing as pdt

from vascpy import PointVasculature, SectionVasculature
from vascpy.cli import app

DATA = Path(__file__).parent / "data"

"""
def test_version():

    from vascpy.version import VERSION

    runner = CliRunner()

    result = runner.invoke(app, "--version")

    assert result.exit_code == 0
    assert result.output == f"vascpy, version {VERSION}\n"
"""


def test_convert__sonata_morphology():

    morphology_file = str(DATA / "h5/fork_diameters_section_spec.h5")
    sonata_file = str(DATA / "h5/fork_diameters_sonata_spec.h5")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:

        out_sonata_file = str(Path(temp_dir) / "sonata.h5")

        with runner.isolated_filesystem(temp_dir=temp_dir):

            result = runner.invoke(app, ["morphology-to-sonata", morphology_file, out_sonata_file])

            assert result.exit_code == 0, result
            assert Path(out_sonata_file).exists()

            v1 = PointVasculature.load_sonata(out_sonata_file)
            v2 = PointVasculature.load_sonata(sonata_file)

            pdt.assert_frame_equal(v1.node_properties, v2.node_properties)
            pdt.assert_frame_equal(v1.edge_properties, v2.edge_properties)

        out_morphology_file = str(Path(temp_dir) / "morphology.h5")

        with runner.isolated_filesystem(temp_dir=temp_dir):

            result = runner.invoke(app, ["sonata-to-morphology", sonata_file, out_morphology_file])

            assert result.exit_code == 0, result.exc_info
            assert Path(out_morphology_file).exists()

            v1 = SectionVasculature.load(morphology_file).as_point_graph()
            v2 = SectionVasculature.load(out_morphology_file).as_point_graph()

            pdt.assert_frame_equal(v1.node_properties, v2.node_properties)
            pdt.assert_frame_equal(v1.edge_properties, v2.edge_properties)
