""" Lazy IO for vtk and h5 loaders and writers """

import logging

import numpy as np

L = logging.getLogger(__name__)


class HDF5:
    """Section hdf5 writer"""

    @staticmethod
    def write(filepath, points, diameters, sections):
        """Save vasculature according to the new spec"""
        import h5py

        n_sections = len(sections)

        structure = np.empty((n_sections, 2), dtype=np.uint64)

        counts = np.empty(len(sections) + 1)
        counts[0] = 0

        connectivity = []

        for i, section in enumerate(sections):

            counts[i + 1] = len(section.points)
            structure[i, 1] = int(section.type)

            for child in section.successors:
                connectivity.append([section.id, child.id])

        structure[:, 0] = np.cumsum(counts)[:-1]
        connectivity = np.asarray(connectivity)
        point_data = np.column_stack((points, diameters))

        with h5py.File(filepath, "w") as fd:
            fd.create_dataset("points", data=point_data)
            fd.create_dataset("structure", data=structure)
            fd.create_dataset("connectivity", data=connectivity)
