"""Curation of point graphs, which is mutative"""

import logging

import numpy as np

from vascpy.utils.adjacency import AdjacencyMatrix

DISTANCE_FACTOR = 10.0

L = logging.getLogger(__name__)


def curate_point_graph(
    point_graph,
    remove_self_loops=False,
    remove_very_long_edges=False,
    remove_high_degree_vertices=False,
    remove_isolated_vertices=False,
):
    """
    Points and edges curation
    """
    points, edges = point_graph.points, point_graph.edges

    edges_to_keep = np.ones(len(edges), dtype=bool)
    vertices_to_keep = np.ones(len(points), dtype=bool)

    if remove_very_long_edges:
        edges_to_keep &= _edges_shorter_than(points, edges, DISTANCE_FACTOR)

    if remove_self_loops:
        edges_to_keep &= _edges_no_self_loops(edges)

    if remove_high_degree_vertices:
        adjacency = AdjacencyMatrix(edges[edges_to_keep], n_vertices=len(points))
        vertices_to_keep &= adjacency.degrees <= 4
        edges_to_keep &= np.all(vertices_to_keep[edges], axis=1)

    if remove_isolated_vertices:
        adjacency = AdjacencyMatrix(edges[edges_to_keep], n_vertices=len(points))
        vertices_to_keep &= adjacency.degrees > 0
        edges_to_keep &= np.all(vertices_to_keep[edges], axis=1)

    point_graph.remove(
        node_indices=np.where(~vertices_to_keep)[0], edge_indices=np.where(~edges_to_keep)[0]
    )


def _edges_shorter_than(points, edges, distance_factor):
    distances = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    return distances < distance_factor * np.median(distances)


def _edges_no_self_loops(edges):
    return edges[:, 0] != edges[:, 1]
