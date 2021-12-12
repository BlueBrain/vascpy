""" Contains features that can be extracted from a vasculature object """

import numpy as np


def segment_lengths(vasc):
    """Returns the distribution of segment lengths of the vasculature object"""
    seg_starts, seg_ends = vasc.points[vasc.edges.T]
    return np.linalg.norm(seg_ends - seg_starts, axis=1)


def segment_volumes(vasc):
    """Returns the distribution of segment volumes of the vasculature object"""
    radii_starts, radii_ends = 0.5 * vasc.diameters[vasc.edges.T]
    seg_lengths = segment_lengths(vasc)

    return (
        (1.0 / 3.0)
        * np.pi
        * (radii_starts ** 2 + radii_starts * radii_ends + radii_ends ** 2)
        * seg_lengths
    )


def segment_slant_heights(vasc):
    """Returns the slant heights of the truncated cone segments"""
    radii_starts, radii_ends = 0.5 * vasc.diameters[vasc.edges.T]

    seg_lengths = segment_lengths(vasc)

    return np.sqrt(seg_lengths ** 2 + (radii_ends - radii_starts) ** 2)


def segment_lateral_areas(vasc):
    """Returns the lateral areas of the trunacted cone segments"""
    radii_starts, radii_ends = 0.5 * vasc.diameters[vasc.edges.T]

    slant_heights = segment_slant_heights(vasc)

    return np.pi * (radii_starts + radii_ends) * slant_heights
