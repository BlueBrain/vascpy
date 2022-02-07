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
# pylint: disable=import-outside-toplevel, too-few-public-methods
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

        with h5py.File(filepath, "w") as file_object:
            file_object.create_dataset("points", data=point_data)
            file_object.create_dataset("structure", data=structure)
            file_object.create_dataset("connectivity", data=connectivity)
