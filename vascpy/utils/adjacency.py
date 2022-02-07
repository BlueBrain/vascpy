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
import numpy as np
from scipy import sparse


def create_sparse_matrix(edges, shape, weights=None):
    """Returns a sparse matrix constructed via the edges list
    Args:
        edges:
            Edge list

        n_vertices:
            The total number of vertices

        weights:
            Edge weights. If None 1 is assigned instead
    """
    if weights is None:
        weights = np.ones(len(edges), dtype=np.int64)

    return sparse.csr_matrix((weights, edges.T.astype(np.int64)), shape=shape, dtype=weights.dtype)


def _return_indices(mask):
    return np.where(mask)[0]


class GraphMatrix:
    """Graph represented using a sparse matrix and its transpose"""

    def __init__(self, edges, shape, weights=None):

        self.shape = shape
        self._weights = weights

        # keep track of the input edge ids
        self._edge_indices = np.lexsort((edges[:, 1], edges[:, 0]))

        # sparse matrix to extract its internal sparse representation
        s_matrix = create_sparse_matrix(edges, shape, weights)

        # we don't need to keep the sparse matrix data structure
        # jus its indexing arrays
        self._indptr = s_matrix.indptr
        self._indices = s_matrix.indices

        # transposed csr matrix for accessing its columns
        s_matrix_transpose = s_matrix.T.tocsr()
        self._indptr_transpose = s_matrix_transpose.indptr
        self._indices_transpose = s_matrix_transpose.indices

    def _row_indices(self, row_index):
        return self._indices[self._indptr[row_index] : self._indptr[row_index + 1]]

    def _col_indices(self, col_index):
        return self._indices_transpose[
            self._indptr_transpose[col_index] : self._indptr_transpose[col_index + 1]
        ]

    @property
    def n_edges(self):
        """Returns the number of edges stored in the adjacency"""
        return self._indptr[-1]

    def as_sparse(self):
        """Returns a sparse matrix representation of the adjacency"""
        # indptr hols the offsets to index data, thus its last elements
        # is size of the data
        data = np.ones(self.n_edges, dtype=np.int64) if self._weights is None else self._weights
        return sparse.csr_matrix(
            (data, self._indices, self._indptr), shape=self.shape, dtype=np.int64
        )

    def edge_index(self, row_index, col_index):
        """Get the edge index for (vertex1, vertex2)"""
        # Get column indices of occupied values
        index_start = self._indptr[row_index]
        index_end = self._indptr[row_index + 1]

        # contains indices of occupied cells at a specific row
        row_indices = self._indices[index_start:index_end]

        # Find a positional index for a specific column index
        ids_array = np.where(row_indices == col_index)[0]

        if ids_array.size > 0:
            local_pos = ids_array[0]
            return self._edge_indices[index_start + local_pos]

        # non-zero value is not found
        return -1

    def predecessors(self, vertex):
        """Get the parents of a given vertex."""
        return self._col_indices(vertex)

    def successors(self, vertex):
        """Get the children of a given vertex."""
        return self._row_indices(vertex)


class AdjacencyMatrix(GraphMatrix):
    """Adjacency Matrix"""

    def __init__(self, edges, n_vertices=None, weights=None):
        self.n_vertices = edges.max() + 1 if n_vertices is None else n_vertices
        super().__init__(edges, (self.n_vertices, self.n_vertices), weights)

    def number_of_self_loops(self):
        """Vertices that loop to themselves"""
        return np.count_nonzero(self.as_sparse().diagonal())

    @property
    def outdegrees(self):
        """
        Summing the adjancency matrix over the columns returns the number of edges that come out
        from each vertec.
        """
        return np.diff(self._indptr)

    @property
    def indegrees(self):
        """
        Summing the adjacency matrix over the rows returns the number of edges that come in each
        vertex.
        """
        return np.diff(self._indptr_transpose)

    @property
    def degrees(self):
        """The degree of each vertex is the sum of all the incoming and outcoming edges."""
        return self.indegrees + self.outdegrees

    def sources(self):
        """Vertices that have indegree 0"""
        mask = (self.indegrees == 0) & (self.outdegrees > 0)
        return _return_indices(mask)

    def sinks(self):
        """Vertices that have outdegree 0"""
        mask = (self.outdegrees == 0) & (self.indegrees > 0)
        return _return_indices(mask)

    def terminations(self):
        """Vertices connected to only one other vertex"""
        mask = self.degrees == 1
        return _return_indices(mask)

    def continuations(self):
        """Vertices that have one parent and one child"""
        mask = (self.indegrees == 1) & (self.outdegrees == 1)
        return _return_indices(mask)

    def isolated_vertices(self):
        """Non connected vertices"""
        mask = self.degrees == 0
        return _return_indices(mask)

    def neighbors(self, vertex_index):
        """Get all adjacent vertices to the given vertex."""
        return np.union1d(self._col_indices(vertex_index), self._row_indices(vertex_index))

    def connected_components(self, directed=False):
        """Connected components of adjacency"""
        # number of components and component label array
        n_components, labels = sparse.csgraph.connected_components(
            self.as_sparse(), return_labels=True, directed=directed
        )

        _, counts = np.unique(labels, return_counts=True)

        offsets = np.empty(n_components + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:] = np.cumsum(counts)

        return np.argsort(labels), offsets


class IncidenceMatrix(GraphMatrix):
    """Incidence matrix representing the connectivity between edges and vertices"""

    def __init__(self, edges, n_vertices):

        incidence_edges = np.empty((2 * len(edges), 2), dtype=edges.dtype)

        incidence_edges[:, 0] = edges.ravel()
        incidence_edges[:, 1] = np.repeat(np.arange(len(edges), dtype=edges.dtype), 2)

        weights = np.tile([-1, 1], len(edges))

        shape = (n_vertices, len(edges))
        super().__init__(incidence_edges, shape, weights)

    def incident(self, vertex):
        """Returns incident edges to vertex"""
        return self._row_indices(vertex)
