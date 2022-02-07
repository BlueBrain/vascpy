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

import click

from vascpy import PointVasculature, SectionVasculature

L = logging.getLogger("vascpy")


FILE_TYPE = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)


@click.group("vascpy", help=__doc__.format(esc="\b"))
@click.option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
def app(verbose=0):
    """VasculatureAPI"""
    # pylint: disable=missing-docstring
    level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }[verbose]

    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.WARNING,
    )
    L.setLevel(level)


@app.command()
@click.argument("input-file", type=FILE_TYPE)
@click.argument("output-file", type=str)
def morphology_to_sonata(input_file, output_file):
    """Converts a section graph morphology to a sonata node population"""
    SectionVasculature.load(input_file).as_point_graph().save_sonata(output_file)


@app.command()
@click.argument("input-file", type=FILE_TYPE)
@click.argument("output-file", type=str)
def sonata_to_morphology(input_file, output_file):
    """Convert sonata file to morphology geometry file"""
    PointVasculature.load_sonata(input_file).as_section_graph().save(output_file)
