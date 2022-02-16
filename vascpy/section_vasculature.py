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
import morphio.vasculature

import vascpy.conversion
import vascpy.section_graph.io


class SectionVasculature:
    """Section representation of graphs"""

    def __init__(self, filepath):
        self._graph = morphio.vasculature.Vasculature(filepath)

    @classmethod
    def load(cls, filepath):
        """Load morphology from file"""
        return cls(filepath)

    @property
    def points(self):
        """Returns the coordinates of the vertices in the morphology"""
        return self._graph.points

    @property
    def diameters(self):
        """Returns all diameters of the vertices in the morphology"""
        return self._graph.diameters

    @property
    def sections(self):
        """Returns the sections of the morphology"""
        return self._graph.sections

    def iter(self):
        """Returns and iterator on the vasculature sections"""
        return self._graph.iter()

    def as_point_graph(self):
        """Creates a point graph representation of the section graph
        Args:
            mutable:
                If True it will return a mutable point graph
        Returns:
            Point graph
        """
        return vascpy.conversion.convert_section_to_point_vasculature(self)

    def save(self, filepath):
        """Write morphology to file"""
        vascpy.section_graph.io.HDF5.write(filepath, self.points, self.diameters, self.sections)
