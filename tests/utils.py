import tempfile

from vascpy.section_graph.io import save_section_graph_data


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


def build_section_morphology(sections, connectivity):

    with tempfile.NamedTemporaryFile(suffix=".h5") as tfile:

        temp_filepath = tfile.name

        save_section_graph_data(
            temp_filepath, chains, chain_connectivity, point_data, section_types
        )

        return SectionVasculature.load(temp_filepath)
