""" Vasculature representation as a Section Graph """
from morphio.vasculature import Vasculature

from vascpy.section_graph import io


class SectionVasculature:
    """Section representation of graphs"""

    def __init__(self, filepath):
        self._graph = Vasculature(filepath)

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
        from vascpy.conversion import convert_section_to_point_vasculature

        return convert_section_to_point_vasculature(self)

    def save(self, filepath):
        """Write morphology to file"""
        io.HDF5.write(filepath, self.points, self.diameters, self.sections)
