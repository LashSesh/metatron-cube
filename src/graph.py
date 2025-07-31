"""
graph.py
---------

This module implements the core data structure representing the
Metatron Cube as a graph.  Each graph consists of a set of canonical
nodes (imported from :mod:`geometry`) and an undirected edge list.  The
class :class:`MetatronCubeGraph` provides methods to obtain the
adjacency matrix, to add or remove edges, and to apply permutations on
the node order (via the group‑theoretic operators defined in
:mod:`symmetries`).

While the default instance uses the canonical nodes and partial edge
list from the blueprint【256449862750268†L1183-L1194】, the class accepts custom
node/edge inputs for experimentation.  This allows integration with
additional research modules (e.g. QLogic, Tensor networks) without
changing the core definitions.

The adjacency matrix is always symmetric, reflecting the undirected
nature of the Metatron Cube.  Self‑loops are not used and are
explicitly prohibited.

Example
-------

>>> from metatron_cube.src.graph import MetatronCubeGraph
>>> g = MetatronCubeGraph()
>>> A = g.get_adjacency_matrix()
>>> A.shape
(13, 13)
>>> g.degree(1)  # degree of the centre node
6
>>> perm = (1, 3, 2, 4, 5, 6, 7)  # simple swap of H1 and H2
>>> g2 = g.permute(perm)
>>> np.allclose(g2.get_adjacency_matrix(), A[[0,2,1,3,4,5,6,7,8,9,10,11,12],:][:,[0,2,1,3,4,5,6,7,8,9,10,11,12]])
True

"""

from __future__ import annotations

from typing import List, Tuple, Iterable, Optional, Dict
import numpy as np

from .geometry import canonical_nodes, canonical_edges, Node

class MetatronCubeGraph:
    """A graph representation of the Metatron Cube.

    Parameters
    ----------
    nodes : Iterable[Node], optional
        The nodes to use.  If not provided, the canonical nodes from
        :func:`geometry.canonical_nodes` are used.  Node indices must be
        consecutive and 1‑based.
    edges : Iterable[Tuple[int, int]], optional
        A list of undirected edges given as pairs of node indices (1‑based).
        If omitted, the canonical edge list from
        :func:`geometry.canonical_edges` is used.
    """

    def __init__(self, nodes: Optional[Iterable[Node]] = None,
                 edges: Optional[Iterable[Tuple[int, int]]] = None,
                 weighted_edges: Optional[Iterable[Tuple[int, int, float]]] = None) -> None:
        """
        Create a new Metatron Cube graph.  Nodes default to the canonical
        13 nodes if not provided.  Edges may be given as a list of pairs
        ``(i, j)`` for unweighted graphs or as triples ``(i, j, w)`` for
        weighted graphs.  If neither ``edges`` nor ``weighted_edges`` is
        provided, the canonical edge list will be used.

        Parameters
        ----------
        nodes : Iterable[Node], optional
            Custom node objects to use.  Indices must be consecutive and
            1‑based.
        edges : Iterable[Tuple[int, int]], optional
            Undirected edges with implicit weight 1.  Ignored if
            ``weighted_edges`` is provided.
        weighted_edges : Iterable[Tuple[int, int, float]], optional
            Undirected edges with explicit weight.  Each element must be a
            triple ``(i, j, w)`` where ``w`` is a real number.
        """
        # Use canonical definitions if arguments not provided
        self.nodes: List[Node] = list(nodes) if nodes is not None else canonical_nodes()
        # Validate node indices are unique and consecutive
        indices = [n.index for n in self.nodes]
        if sorted(indices) != list(range(1, len(self.nodes) + 1)):
            raise ValueError("Node indices must be unique, 1‑based and consecutive")

        # Build the internal edge list with weights
        self.edge_weights: Dict[Tuple[int, int], float] = {}
        # If weighted edges are provided, ignore unweighted edges
        if weighted_edges is not None:
            for (i, j, w) in weighted_edges:
                if i == j:
                    raise ValueError(f"Self‑loops are not allowed: edge ({i}, {j})")
                key = (min(i, j), max(i, j))
                self.edge_weights[key] = float(w)
        else:
            # Fallback to unweighted edges (canonical if None)
            if edges is None:
                edges = canonical_edges()
            for (i, j) in edges:
                if i == j:
                    raise ValueError(f"Self‑loops are not allowed: edge ({i}, {j})")
                key = (min(int(i), int(j)), max(int(i), int(j)))
                # assign weight 1 if not specified
                self.edge_weights[key] = 1.0

        # Precompute adjacency matrix with weights
        self._adjacency: np.ndarray = self._compute_adjacency_matrix()

    # ------------------------------------------------------------------
    # Basic graph construction
    # ------------------------------------------------------------------
    def _compute_adjacency_matrix(self) -> np.ndarray:
        """
        Compute the symmetric adjacency matrix from the current edge list.

        Returns
        -------
        numpy.ndarray
            A square matrix where ``A[i,j]`` is the weight of the edge
            between node ``i+1`` and ``j+1``.  If the graph is unweighted
            this will be 0 or 1.  The matrix is symmetric.
        """
        n = len(self.nodes)
        # use float dtype to allow weighted edges
        A = np.zeros((n, n), dtype=float)
        for (i, j), w in self.edge_weights.items():
            u = i - 1
            v = j - 1
            A[u, v] = w
            A[v, u] = w
        return A

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return a copy of the adjacency matrix.

        Returns
        -------
        numpy.ndarray
            A binary symmetric matrix of shape (n, n) where ``n`` is the
            number of nodes.
        """
        return self._adjacency.copy()

    # ------------------------------------------------------------------
    # Node and edge utilities
    # ------------------------------------------------------------------
    def neighbors(self, index: int) -> List[int]:
        """Return the neighboring node indices of a given node.

        Parameters
        ----------
        index : int
            1‑based node index.

        Returns
        -------
        List[int]
            List of indices adjacent to ``index``.
        """
        self._validate_node_index(index)
        idx0 = index - 1
        row = self._adjacency[idx0]
        return [i + 1 for i, v in enumerate(row) if v == 1]

    def degree(self, index: int) -> int:
        """Return the degree (number of incident edges) of a node."""
        return len(self.neighbors(index))

    def add_edge(self, i: int, j: int) -> None:
        """Add an undirected edge between nodes ``i`` and ``j``.

        Duplicate edges are ignored.  Self‑loops are not permitted.
        After modification, the adjacency matrix is recomputed.
        """
        self.add_weighted_edge(i, j, 1.0)

    def add_weighted_edge(self, i: int, j: int, weight: float) -> None:
        """Add or update an undirected edge with the given weight.

        Parameters
        ----------
        i, j : int
            1‑based node indices.
        weight : float
            Weight for the edge.  If zero, the edge is removed.
        """
        self._validate_node_index(i)
        self._validate_node_index(j)
        if i == j:
            raise ValueError("Self‑loops are not allowed")
        key = (min(i, j), max(i, j))
        if weight == 0:
            if key in self.edge_weights:
                del self.edge_weights[key]
        else:
            self.edge_weights[key] = float(weight)
        self._adjacency = self._compute_adjacency_matrix()

    def remove_edge(self, i: int, j: int) -> None:
        """Remove the undirected edge between ``i`` and ``j`` if present."""
        key = (min(i, j), max(i, j))
        if key in self.edge_weights:
            del self.edge_weights[key]
            self._adjacency = self._compute_adjacency_matrix()

    def _validate_node_index(self, index: int) -> None:
        if index < 1 or index > len(self.nodes):
            raise IndexError(f"Node index {index} out of bounds (1..{len(self.nodes)})")

    # ------------------------------------------------------------------
    # Permutations and symmetries
    # ------------------------------------------------------------------
    def permute(self, sigma: Tuple[int, ...]) -> 'MetatronCubeGraph':
        """Return a new graph with node order permuted by ``sigma``.

        The permutation ``sigma`` must be a tuple of length equal to the
        number of nodes, containing each integer from 1..n exactly once.
        It describes the new order of the nodes (1‑based).  For example,
        ``sigma = (1, 3, 2, 4, ...)`` swaps nodes 2 and 3.

        The method remaps both the nodes and the edge indices.  It does
        *not* change the underlying geometry; only the labels/indices are
        permuted.  Use this to explore isomorphic graphs under S7 or
        subgroup actions.【256449862750268†L1460-L1466】

        Parameters
        ----------
        sigma : Tuple[int, ...]
            A permutation of ``(1..n)``.

        Returns
        -------
        MetatronCubeGraph
            A new graph with permuted nodes and edges.
        """
        n = len(self.nodes)
        if len(sigma) != n or set(sigma) != set(range(1, n + 1)):
            raise ValueError("sigma must be a permutation of 1..n")
        # Permute the nodes
        new_nodes = [self.nodes[i - 1] for i in sigma]
        # Remap edges by applying sigma to each endpoint
        idx_map = {old: new_idx + 1 for new_idx, old in enumerate(sigma)}
        new_edge_weights: Dict[Tuple[int, int], float] = {}
        for (i, j), w in self.edge_weights.items():
            new_i = idx_map[i]
            new_j = idx_map[j]
            key = (min(new_i, new_j), max(new_i, new_j))
            new_edge_weights[key] = w
        return MetatronCubeGraph(nodes=new_nodes, weighted_edges=[(k[0], k[1], w) for k, w in new_edge_weights.items()])

    def apply_permutation_matrix(self, P: np.ndarray) -> 'MetatronCubeGraph':
        """Apply a 13×13 permutation matrix to the adjacency matrix.

        This method is a thin wrapper around matrix multiplication
        ``A′ = P A Pᵀ`` as defined in the blueprint【256449862750268†L1460-L1466】.  It produces a
        new graph with the same node order but with adjacency corresponding
        to the permuted indices.  The provided matrix ``P`` must be
        orthogonal and binary (a valid permutation matrix).

        Parameters
        ----------
        P : numpy.ndarray
            Permutation matrix of shape (n, n).

        Returns
        -------
        MetatronCubeGraph
            A new graph whose adjacency matrix equals ``P @ A @ P.T``.
        """
        A = self.get_adjacency_matrix()
        if P.shape != A.shape:
            raise ValueError("Permutation matrix must be same shape as adjacency matrix")
        # Compute the new adjacency matrix
        A_prime = P @ A @ P.T
        # Derive new edge list from A_prime
        n = len(self.nodes)
        new_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A_prime[i, j] != 0:
                    new_edges.append((i + 1, j + 1))
        return MetatronCubeGraph(nodes=self.nodes, edges=new_edges)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"MetatronCubeGraph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"

    def __str__(self) -> str:
        return f"MetatronCubeGraph with {len(self.nodes)} nodes and {len(self.edges)} edges"
