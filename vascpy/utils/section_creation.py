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
from collections import defaultdict, deque

import numpy as np

from vascpy.utils.adjacency import AdjacencyMatrix

L = logging.getLogger(__name__)


def create_chains(edges, n_points, return_index=False):
    """Generates chains of nodes of flow_through (not branching) vertices from the set
    of the provides edges starting the traversal at start_node.

    Returns an array of the sections, their map to the original edges and the connectivity
    among the sections (as a whole, not among edges)
    """
    # avoid having symmetric edges e.g. [1, 2] and [2, 1] by sorting the columns
    edges = np.sort(edges, 1)

    # create adjacnecy from edges
    adjacency = AdjacencyMatrix(edges)

    L.info("Extracting chain structure...")
    # get the structure of each section
    chains = _chain_structure(adjacency, edges, n_points)

    L.info("Creating section connectivity...")

    # get the connectivity among the sections
    chain_connectivity = _chain_connectivity(edges, chains)

    if return_index:
        index_chains = _map_chains_to_original_edges(edges, chains)
        return chains, chain_connectivity, index_chains

    return chains, chain_connectivity


def reconstruct_chains_using_groups(edges, group_ids, return_index=True):
    """Reconstruct section using the group ids"""

    edge_ids = np.argsort(group_ids, kind="stable")

    _, idx, counts = np.unique(group_ids, return_index=True, return_counts=True)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)

    edge_ids_per_chain = [
        edge_ids[offsets[i] : offsets[i + 1]] for i in np.argsort(idx, kind="stable")
    ]
    edges_per_chain = [edges[edge_ids] for edge_ids in edge_ids_per_chain]

    # get the connectivity among the sections
    chain_connectivity = _chain_connectivity(edges, edges_per_chain)

    if return_index:
        return edges_per_chain, chain_connectivity, edge_ids_per_chain

    return edges_per_chain, chain_connectivity


def _is_flow_through(indeg, outdeg):
    return (
        ((indeg == 1) & (outdeg == 1))
        | ((indeg == 0) & (outdeg == 1))
        | ((indeg == 1) & (outdeg == 0))
        | ((indeg == 2) & (outdeg == 0))
        | ((indeg == 0) & (outdeg == 2))
    )


def _component_start_nodes(adjacency, deg):
    """Find first nodes of components"""
    # pylint: disable=too-many-locals

    vertices, offsets = adjacency.connected_components()
    comp_start_nodes = []

    n_fork_components = 0
    n_cont_components = 0
    n_term_components = 0

    n_one_node_components = 0
    n_self_reference_nodes = 0

    for i in range(len(offsets) - 1):

        nodes = vertices[offsets[i] : offsets[i + 1]]
        n_nodes = len(nodes)

        # isolated vertices, ignore them
        if n_nodes == 1:
            n_one_node_components += 1
            continue

        # two identical nodes, not much use of its
        if n_nodes == 2 and nodes[0] == nodes[1]:
            n_self_reference_nodes += 1
            continue

        # we prefer to start from a termination
        terminations = nodes[deg[nodes] == 1]

        if terminations.size > 0:
            start_node = terminations.min()
            comp_start_nodes.append(start_node)
            n_term_components += 1
            continue

        # if no terminations, check for forks
        forks = nodes[deg[nodes] > 2]

        if len(forks) > 0:
            start_node = forks.min()
            comp_start_nodes.append(start_node)
            n_fork_components += 1
            continue

        # if nothing else use a continuation
        start_node = nodes.min()
        n_cont_components += 1
        comp_start_nodes.append(start_node)

    L.info("Found %d self-reference edges, eg [0, 0]", n_self_reference_nodes)
    L.info("Found %d one node component that were ignored.", n_one_node_components)
    L.info("Found %d components with a fork for start_node", n_fork_components)
    L.info("Found %d components with a cont for start_node", n_cont_components)
    L.info("Found %d components with a term for start_node", n_term_components)
    L.info("Number of Connected Components: %s", len(comp_start_nodes))

    return comp_start_nodes


def _add_to_section(current_section, beg, end):
    """Expand chain"""
    if current_section:

        queue_beg = current_section[0][0]
        queue_end = current_section[-1][-1]

        if beg == queue_end:
            current_section.append((beg, end))
        elif beg == queue_beg:
            current_section.appendleft((end, beg))
        elif end == queue_end:
            current_section.append((end, beg))
        elif end == queue_beg:
            current_section.appendleft((beg, end))
        else:
            msg = f"Unconnected edge: \n{current_section} <-- ({beg}, {end})\n"
            # msg += 'current_node: {}, deg: {}'.format(beg, degree[beg])
            # raise ValueError(msg)
            L.info(msg)
    else:
        current_section.append((beg, end))


def _exhaust_chain(adjacency, initial_node, neighbor, visited, is_chain):
    """Traverse chain"""

    current_section = deque()

    _add_to_section(current_section, initial_node, neighbor)

    chain_nodes = set([initial_node, neighbor])

    # single edge section
    if not is_chain[neighbor]:
        return neighbor, current_section

    queue = deque([neighbor])

    initial_node_occurence = 0

    is_first_time = True

    while queue:

        current_node = queue.pop()

        for next_node in adjacency.neighbors(current_node):

            # count the number of times you see
            # the initial_node via the neighbors
            if next_node == initial_node:
                initial_node_occurence += 1

            # if it is encountered twice, it means that we have a loop
            if initial_node_occurence == 2 and is_first_time:
                is_first_time = False
                _add_to_section(current_section, current_node, initial_node)

            if next_node not in chain_nodes:

                visited[next_node] = True
                chain_nodes.add(next_node)

                _add_to_section(current_section, current_node, next_node)

                if is_chain[next_node]:
                    queue.append(next_node)
                else:
                    return next_node, current_section
    return None, current_section


def _chain_structure(adjacency, edges, n_points):
    """Given edges and their adjacency create sections"""
    degree = adjacency.degrees

    is_chain = (degree > 0) & (degree <= 2)

    sections = [deque() for _ in range(edges.shape[0])]
    sections = []

    visited = np.zeros(n_points, dtype=bool)

    queue = deque()

    for start_node in _component_start_nodes(adjacency, degree):

        queue.append(start_node)
        visited[start_node] = True

        while queue:

            current_node = queue.pop()

            for neighbor in adjacency.neighbors(current_node):

                if not visited[neighbor]:

                    visited[neighbor] = True

                    end_node, section = _exhaust_chain(
                        adjacency, current_node, neighbor, visited, is_chain
                    )

                    sections.append(section)

                    if end_node is not None:
                        assert not degree[end_node] == 2, degree[end_node]
                        queue.append(end_node)

    return sections


def _chain_connectivity(edges, chains):
    """Returns chain connectivity treated as clustered entities represented by nodes"""
    chain_connectivity = np.empty((len(edges), 2), dtype=np.int64)

    starts = defaultdict(list)
    for section_index, chain in enumerate(chains):
        starts[chain[0][0]].append(section_index)

    i_connection = 0
    for section_index, chain in enumerate(chains):

        section_end = chain[-1][1]

        for child_section in starts[section_end]:
            if section_index != child_section:

                chain_connectivity[i_connection, 0] = section_index
                chain_connectivity[i_connection, 1] = child_section
                i_connection += 1

    return chain_connectivity[:i_connection]


def _map_chains_to_original_edges(edges, chains):
    """Returns the indices of the edges in the chains as they found in edges array.
    Directionality is not kept into account, i.e. (0, 1) and (1, 0) will map at the
    same index in edges.
    """
    # frozen set allows for undirected edges
    edge_map = {frozenset(edge): i for i, edge in enumerate(edges)}
    return [
        np.fromiter((edge_map[frozenset(edge)] for edge in chain), dtype=np.int64)
        for chain in chains
    ]
