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
# pylint: disable=too-few-public-methods
import typing

import numpy as np


class _SpecType(typing.NamedTuple):
    """Specification type dataclass"""

    name: str
    dtype: type


class SpecPointVasculature:
    """PointVasculature specification"""

    # pylint: disable=invalid-name, too-many-instance-attributes

    X: _SpecType = _SpecType("x", np.float32)
    Y: _SpecType = _SpecType("y", np.float32)
    Z: _SpecType = _SpecType("z", np.float32)
    D: _SpecType = _SpecType("diameter", np.float32)

    NODE_PROPERTIES: typing.Tuple[_SpecType, ...] = (X, Y, Z, D)

    SOURCE: _SpecType = _SpecType("start_node", np.int64)
    TARGET: _SpecType = _SpecType("end_node", np.int64)
    TYPE: _SpecType = _SpecType("type", np.int32)
    SECTION_ID: _SpecType = _SpecType("section_id", np.int32)
    SEGMENT_ID: _SpecType = _SpecType("segment_id", np.int32)

    EDGE_PROPERTIES: typing.Tuple[_SpecType, ...] = (SOURCE, TARGET, TYPE, SECTION_ID, SEGMENT_ID)


class SpecSectionVasculature:
    """SectionVasculature specification"""

    # pylint: disable=invalid-name

    POINTS: type = np.float32
    STRUCTURE: type = np.int64
    CONNECTIVITY: type = np.int64


class SpecSectionHDF5:
    """SectionVasculature file format specification"""

    # pylint: disable=invalid-name

    POINTS: type = np.float32
    STRUCTURE: type = np.int64
    CONNECTIVITY: type = np.int64


class SpecSONATA:
    """SONATA file format specification"""

    # pylint: disable=invalid-name

    SONATA_POINT_DTYPES: typing.Dict[str, type] = {
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "diameter": np.float32,
    }

    SONATA_EDGE_DTYPES: typing.Dict[str, type] = {
        "type": np.int32,
        "start_node": np.uint64,
        "end_node": np.uint64,
        "section_id": np.uint32,
        "segment_id": np.uint32,
    }
