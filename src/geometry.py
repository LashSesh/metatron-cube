"""
geometry.py
--------------

This module provides the canonical geometric definition of the Metatron Cube
as described in the blueprint.  It defines the 13 nodes (one centre, six
hexagon vertices and six cube corners) and their explicit 3D coordinates.
It also exposes helper functions for working with these nodes and basic
distance calculations.

The aim of this module is twofold:

* Provide a single source of truth for the canonical node list used by
  :class:`~metatron_cube.src.graph.MetatronCubeGraph` and related modules.
* Offer simple utility functions such as computing pairwise distances or
  looking up nodes by label or index.

The node coordinates follow the convention laid out in the "Complete
Canonical Node Table" of the blueprint【256449862750268†L940-L1034】.  The six cube
corners deliberately omit the two negative–negative combinations to match
the 13‑node structure used throughout the prototype.

This file is intentionally free of any application logic; it merely
describes the geometry of the static graph.  Should alternative
coordinate systems or node labellings be required, you can extend
``canonical_nodes`` or write your own helper functions accordingly.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import numpy as np


@dataclass(frozen=True)
class Node:
    """Represents a single node in the Metatron Cube.

    Attributes
    ----------
    index : int
        1‑based index of the node as defined in the canonical list.
    label : str
        Symbolic name of the node (e.g. ``"C"``, ``"H1"``, ``"Q1"``).
    type : str
        High‑level category of the node (``"center"``, ``"hexagon"`` or
        ``"cube"``).
    coords : Tuple[float, float, float]
        Cartesian coordinates of the node in ℝ³.
    """

    index: int
    label: str
    type: str
    coords: Tuple[float, float, float]

    def as_array(self) -> np.ndarray:
        """Return the coordinates as a NumPy array.

        Returns
        -------
        numpy.ndarray
            1‑D array containing (x, y, z).
        """
        return np.array(self.coords, dtype=np.float64)

    def distance_to(self, other: 'Node') -> float:
        """Compute the Euclidean distance to another node.

        Parameters
        ----------
        other : Node
            The node to measure the distance to.

        Returns
        -------
        float
            The Euclidean distance between ``self`` and ``other``.
        """
        return float(np.linalg.norm(self.as_array() - other.as_array()))


def canonical_nodes() -> List[Node]:
    """Return the canonical list of Metatron Cube nodes.

    The 13 nodes are defined exactly as in Table 2 of the blueprint【256449862750268†L940-L1034】.
    Node indices start at 1.  For convenience, the coordinates are given as
    plain Python tuples, but all arithmetic is performed using ``numpy``.

    Returns
    -------
    List[Node]
        A list of :class:`Node` instances in ascending index order.
    """
    # Precompute the square root of three for compactness
    sqrt3 = np.sqrt(3.0)
    return [
        Node(1, "C", "center", (0.0, 0.0, 0.0)),
        # Hexagon vertices around the centre in the xy‑plane
        Node(2, "H1", "hexagon", (1.0, 0.0, 0.0)),
        Node(3, "H2", "hexagon", (0.5,  sqrt3 / 2.0, 0.0)),
        Node(4, "H3", "hexagon", (-0.5, sqrt3 / 2.0, 0.0)),
        Node(5, "H4", "hexagon", (-1.0, 0.0, 0.0)),
        Node(6, "H5", "hexagon", (-0.5, -sqrt3 / 2.0, 0.0)),
        Node(7, "H6", "hexagon", (0.5, -sqrt3 / 2.0, 0.0)),
        # Cube corners (a subset of the full cube; missing two negative–negative
        # corners by design)
        Node(8,  "Q1", "cube", (0.5,  0.5,  0.5)),
        Node(9,  "Q2", "cube", (0.5,  0.5, -0.5)),
        Node(10, "Q3", "cube", (0.5, -0.5,  0.5)),
        Node(11, "Q4", "cube", (0.5, -0.5, -0.5)),
        Node(12, "Q5", "cube", (-0.5, 0.5,  0.5)),
        Node(13, "Q6", "cube", (-0.5, 0.5, -0.5)),
    ]


# Alias functions for backwards compatibility
# -----------------------------------------
# A number of legacy tests and scripts refer to ``get_metatron_nodes`` and
# ``get_metatron_edges`` to retrieve the canonical node and edge lists.
# Earlier revisions of this prototype exposed these names.  To avoid
# breaking existing code, the following convenience wrappers simply
# delegate to :func:`canonical_nodes` and :func:`canonical_edges`.  You
# should prefer calling the canonical functions directly in new code.

def get_metatron_nodes() -> List[Node]:
    """Return the canonical list of Metatron Cube nodes.

    This function is an alias for :func:`canonical_nodes` provided for
    backwards compatibility with earlier versions of the prototype and
    existing unit tests.  It returns the same 13 :class:`Node` objects in
    ascending index order.

    Returns
    -------
    List[Node]
        The canonical nodes of the Metatron Cube.
    """
    return canonical_nodes()


def get_metatron_edges(full: bool = False) -> List[Tuple[int, int]]:
    """Return the canonical edge list for the Metatron Cube.

    This function is an alias for :func:`canonical_edges`.  If the
    optional argument ``full`` is set to ``True``, it will instead return
    the exhaustive edge list via :func:`complete_canonical_edges`.

    Parameters
    ----------
    full : bool, optional
        Whether to return all \(13 choose 2\) edges (``True``) or only the
        partial canonical edges (``False``).  Defaults to ``False``.

    Returns
    -------
    List[Tuple[int, int]]
        A list of undirected edge pairs.  See
        :func:`canonical_edges` and :func:`complete_canonical_edges` for
        details.
    """
    return complete_canonical_edges() if full else canonical_edges()


def canonical_edges() -> List[Tuple[int, int]]:
    """Return the canonical edge list for the Metatron Cube.

    The returned list contains pairs of node indices (1‑based) representing
    undirected edges.  It follows the partial enumeration in the blueprint
    (Section 4.6)【256449862750268†L1183-L1194】.  This covers the centre–hexagon edges,
    the hexagon cycle, and a selection of cube edges and diagonals.  Clients
    may extend this list to include additional cross‑connections (e.g.,
    hexagon–cube links, centre–cube edges, or internal platonic solid edges)
    as required.  For a fully connected graph, see :func:`complete_canonical_edges`.

    Returns
    -------
    List[Tuple[int, int]]
        A list of 2‑tuples ``(i, j)`` where ``i`` and ``j`` are node indices.
    """
    # Base edges from the blueprint: centre to hexagon
    edges: List[Tuple[int, int]] = [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        # Hexagon cycle
        (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2),
        # Cube edges (one face): Q1–Q2–Q4–Q3–Q1
        (8, 9), (9, 11), (11, 10), (10, 8),
        # Additional cube cross‑edges and diagonals
        (8, 12), (9, 13), (10, 12), (11, 13), (12, 13),
        # Face diagonals (explicit duplicates removed by caller if needed)
        (8, 10), (9, 11),
    ]
    return edges


def complete_canonical_edges() -> List[Tuple[int, int]]:
    """Return the exhaustive edge list for the Metatron Cube.

    While :func:`canonical_edges` returns a minimal subset of edges for clarity,
    the full Metatron Cube embeds all lines connecting the 13 nodes such that
    every Platonic solid (tetrahedron, cube, octahedron, dodecahedron and
    icosahedron) can be extracted by selecting appropriate subsets.  This
    function enumerates all \(\binom{13}{2} = 78\) undirected edges, yielding
    a complete graph on 13 nodes (K\_13).  It is provided for convenience
    when maximal connectivity is desired【256449862750268†L1183-L1194】.

    Returns
    -------
    List[Tuple[int, int]]
        All undirected edge pairs ``(i, j)`` with ``1 ≤ i < j ≤ 13``.
    """
    return full_edge_list(13)


def full_edge_list(n: int = 13) -> List[Tuple[int, int]]:
    """Return the complete set of undirected edges for ``n`` nodes.

    This convenience function generates all pairs ``(i, j)`` with
    ``1 <= i < j <= n``.  It can be used to build a fully connected
    Metatron Cube graph when a maximal connectivity is desired (e.g.
    representing all five Platonic solids simultaneously).  Use with
    caution for large ``n``.

    Parameters
    ----------
    n : int, optional
        Number of nodes.  Defaults to 13 for the Metatron Cube.

    Returns
    -------
    List[Tuple[int, int]]
        All ``n(n-1)/2`` undirected edge pairs.
    """
    edges: List[Tuple[int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            edges.append((i, j))
    return edges


def find_node(nodes: Iterable[Node], label: Optional[str] = None,
              index: Optional[int] = None) -> Optional[Node]:
    """Find a node by its label or index.

    Exactly one of ``label`` or ``index`` must be provided.  If a node is
    not found, ``None`` is returned.

    Parameters
    ----------
    nodes : Iterable[Node]
        Iterable of nodes to search.
    label : str, optional
        The label of the desired node.
    index : int, optional
        The 1‑based index of the desired node.

    Returns
    -------
    Optional[Node]
        The matching node or ``None`` if no match is found.
    """
    if (label is None) == (index is None):
        raise ValueError("Specify exactly one of label or index")
    for node in nodes:
        if label is not None and node.label == label:
            return node
        if index is not None and node.index == index:
            return node
    return None
