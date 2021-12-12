"""Specifications for the various vasculature data structures
and file formats.
"""
from collections import namedtuple

import numpy as np


class SpecPointVasculature:
    """PointVasculature specification"""

    _Tp = namedtuple("Type", ["name", "dtype"])

    X = _Tp("x", np.float32)
    Y = _Tp("y", np.float32)
    Z = _Tp("z", np.float32)
    D = _Tp("diameter", np.float32)

    NODE_PROPERTIES = [X, Y, Z, D]

    SOURCE = _Tp("start_node", np.int64)
    TARGET = _Tp("end_node", np.int64)
    TYPE = _Tp("type", np.int32)
    SECTION_ID = _Tp("section_id", np.int32)
    SEGMENT_ID = _Tp("segment_id", np.int32)

    EDGE_PROPERTIES = (SOURCE, TARGET, TYPE, SECTION_ID, SEGMENT_ID)


class SpecSectionVasculature:
    """SectionVasculature specification"""

    POINTS = np.float32
    STRUCTURE = np.int64
    CONNECTIVITY = np.int64


class SpecSectionHDF5:
    """SectionVasculature file format specification"""

    POINTS = np.float32
    STRUCTURE = np.int64
    CONNECTIVITY = np.int64


class SpecSONATA:
    """SONATA file format specification"""

    SONATA_POINT_DTYPES = {
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "diameter": np.float32,
    }

    SONATA_EDGE_DTYPES = {
        "type": np.int32,
        "start_node": np.uint64,
        "end_node": np.uint64,
        "section_id": np.uint32,
        "segment_id": np.uint32,
    }
