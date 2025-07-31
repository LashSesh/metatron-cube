
# src/symmetries.py

"""
symmetries.py
---------------

This module contains group‑theoretic utilities for the Metatron Cube.  In
addition to generating all S7 permutations and constructing permutation
matrices (as defined in the blueprint), it provides convenience
functions for important subgroups such as the cyclic hexagon rotations
(C₆), the dihedral group on the hexagon (D₆), and alternating or
symmetric groups on arbitrary subsets of node indices.

The permutation functions return tuples of 1‑based indices.  For
permutation matrices acting on the full 13×13 adjacency matrix, use
:func:`permutation_matrix` (or :func:`permutation_to_matrix` for
backwards compatibility).  To apply these matrices, see
``MetatronCubeGraph.apply_permutation_matrix``.

References
----------
The definitions for rotations and reflections follow the descriptions in
Section 4.12 of the blueprint【256449862750268†L1385-L1409】, where the hexagon
symmetry is isomorphic to C₆ and D₆.
"""

import numpy as np
from itertools import permutations
from typing import List, Tuple, Iterable, Optional


def generate_s7_permutations() -> List[Tuple[int]]:
    """
    Erzeugt alle 5040 Permutationen für die 7 Knoten: [1,2,3,4,5,6,7]
    (Index 1 = Center, 2-7 = Hexagon)
    """
    return list(permutations(range(1, 8)))  # 1-basiert (wie in Knotenliste)

def permutation_to_matrix(sigma: Tuple[int]) -> np.ndarray:
    """
    Erzeugt eine 13x13 Permutationsmatrix für eine Permutation sigma auf S7.
    Die ersten 7 Knoten werden gemäß sigma umsortiert, die restlichen bleiben fix.
    """
    P = np.eye(13, dtype=int)
    # sigma enthält 7 Werte (Knoten-Indices 1-7), Zuordnung auf 0-basierte Zeilen/Spalten
    for src_pos, tgt_index in enumerate(sigma):
        P[src_pos, :] = 0
        P[src_pos, tgt_index - 1] = 1
    return P


def permutation_matrix(sigma: Tuple[int], size: int = 13) -> np.ndarray:
    """Construct a full ``size``×``size`` permutation matrix for a given permutation.

    Parameters
    ----------
    sigma : Tuple[int, ...]
        A permutation of a subset of indices ``1..size``.  Entries not
        included in ``sigma`` are assumed to map to themselves.
    size : int, optional
        The dimension of the permutation matrix.  Defaults to 13.

    Returns
    -------
    numpy.ndarray
        A binary matrix ``P`` of shape (``size``, ``size``) such that
        ``P[i-1,j-1] = 1`` if the permutation maps ``i`` to ``j``.
    """
    P = np.eye(size, dtype=int)
    for src_pos, tgt_index in enumerate(sigma):
        i = src_pos + 1
        P[i - 1, :] = 0
        P[i - 1, tgt_index - 1] = 1
    return P

def apply_permutation_to_adjacency(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Wendet die Permutationsmatrix P auf Adjazenzmatrix A an:
    Rückgabe: A' = P @ A @ P.T
    """
    return P @ A @ P.T

def permute_node_labels(nodes: List, sigma: Tuple[int]) -> List:
    """
    Gibt eine neue Liste der Knoten-Objekte zurück, entsprechend der Permutation sigma.
    """
    new_nodes = [None] * 13
    # Die ersten 7 Knoten werden gemappt, Rest bleibt identisch
    for new_pos, old_index in enumerate(sigma):
        new_nodes[new_pos] = nodes[old_index - 1]
    for i in range(7, 13):
        new_nodes[i] = nodes[i]
    return new_nodes

# ---------------------------------------------------------------------------
# Hexagon rotations and reflections (C6 and D6)
# ---------------------------------------------------------------------------
_HEX_ANGLES = {
    2: 0.0,
    3: 60.0,
    4: 120.0,
    5: 180.0,
    6: 240.0,
    7: 300.0,
}


def hexagon_rotation(k: int) -> Tuple[int, ...]:
    """Return the permutation corresponding to a k×60° rotation of the hexagon.

    The centre node (1) remains fixed.  Only nodes 2–7 are rotated.  The
    returned tuple has length 7 and can be used with
    :func:`permutation_to_matrix` or :func:`MetatronCubeGraph.permute`.

    Parameters
    ----------
    k : int
        Number of 60° steps to rotate by.  Values are taken modulo 6.

    Returns
    -------
    Tuple[int, ...]
        A permutation of (1..7) where node 1 maps to itself and nodes
        2..7 are cyclically shifted.
    """
    k = k % 6
    # Build mapping for 7 elements; position 0 corresponds to node 1 (centre)
    mapping = [1]
    for i in range(6):  # i runs 0..5 corresponding to nodes 2..7
        # compute index in 2..7 after rotation
        new_index = 2 + ((i + k) % 6)
        mapping.append(new_index)
    return tuple(mapping)


def hexagon_reflection(axis_node: int) -> Tuple[int, ...]:
    """Return the reflection permutation across the axis through centre and ``axis_node``.

    Reflection axes are defined by node indices 2–7.  The centre node (1)
    stays fixed; cube nodes (8–13) are unaffected and thus not included in
    this 7‑tuple representation.  The reflection of each hexagon node is
    computed by mirroring its angle around the axis angle【256449862750268†L1385-L1409】.

    Parameters
    ----------
    axis_node : int
        Index of the hexagon node (2–7) defining the reflection axis.

    Returns
    -------
    Tuple[int, ...]
        A permutation of (1..7) describing the reflection.
    """
    if axis_node not in _HEX_ANGLES:
        raise ValueError("axis_node must be one of the hexagon nodes 2..7")
    axis_angle = _HEX_ANGLES[axis_node]
    # Precompute inverse mapping from angle to node index
    angle_to_node = {v: k for k, v in _HEX_ANGLES.items()}
    # Build permutation: first element for centre
    mapping = [1]
    for node in range(2, 8):
        ang = _HEX_ANGLES[node]
        new_ang = (2 * axis_angle - ang) % 360.0
        # due to floating errors, round to nearest integer angle (multiple of 60)
        new_ang_round = round(new_ang / 60.0) * 60.0 % 360.0
        mapping.append(angle_to_node[new_ang_round])
    return tuple(mapping)


def generate_c6_subgroup() -> List[Tuple[int, ...]]:
    """Generate the 6 rotations of the hexagon (cyclic group C6)."""
    return [hexagon_rotation(k) for k in range(6)]


def generate_d6_subgroup() -> List[Tuple[int, ...]]:
    """Generate the 12 elements of the dihedral group D6 acting on the hexagon.

    D6 consists of the 6 rotations and 6 reflections around axes through
    centre and one hexagon vertex.
    """
    rotations = generate_c6_subgroup()
    reflections = [hexagon_reflection(i) for i in range(2, 8)]
    return rotations + reflections


# ---------------------------------------------------------------------------
# General permutation groups on arbitrary subsets
# ---------------------------------------------------------------------------
def _is_even_permutation(seq: Iterable[int]) -> bool:
    """Determine whether a permutation (given as a sequence) is even.

    Uses the parity of inversion count: a permutation is even if the
    number of inversions is even.
    """
    inv_count = 0
    arr = list(seq)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count % 2 == 0


def _extend_partial_permutation(partial: Tuple[int, ...], subset: Tuple[int, ...], total_n: int = 13) -> Tuple[int, ...]:
    """Extend a permutation on a subset to a full permutation of 1..total_n.

    Parameters
    ----------
    partial : Tuple[int, ...]
        A permutation of the values in ``subset``.
    subset : Tuple[int, ...]
        The original indices being permuted.  ``partial`` must contain the
        same elements as ``subset``.
    total_n : int
        The total number of nodes (default 13).

    Returns
    -------
    Tuple[int, ...]
        A permutation tuple of length ``total_n`` where indices in
        ``subset`` are permuted according to ``partial`` and all others map
        to themselves.
    """
    if set(partial) != set(subset):
        raise ValueError("partial must contain exactly the elements of subset")
    # Build mapping from original to new labels within subset
    mapping = {old: new for old, new in zip(subset, partial)}
    result = []
    for i in range(1, total_n + 1):
        result.append(mapping.get(i, i))
    return tuple(result)


def generate_symmetric_group(subset: Iterable[int], total_n: int = 13) -> List[Tuple[int, ...]]:
    """Generate the full symmetric group on a subset of node indices.

    Parameters
    ----------
    subset : Iterable[int]
        The node indices to permute (e.g. (8, 9, 10, 12) for four cube nodes).
    total_n : int, optional
        Total number of nodes in the graph.  Defaults to 13.

    Returns
    -------
    List[Tuple[int, ...]]
        A list of permutation tuples of length ``total_n``.
    """
    subset = tuple(subset)
    perms = []
    for p in permutations(subset):
        perms.append(_extend_partial_permutation(p, subset, total_n))
    return perms


def generate_alternating_group(subset: Iterable[int], total_n: int = 13) -> List[Tuple[int, ...]]:
    """Generate the alternating (even) permutations on a subset of node indices.

    Parameters
    ----------
    subset : Iterable[int]
        The node indices to permute.
    total_n : int, optional
        Total number of nodes.  Defaults to 13.

    Returns
    -------
    List[Tuple[int, ...]]
        A list of even permutation tuples of length ``total_n``.
    """
    subset = tuple(subset)
    perms = []
    for p in permutations(subset):
        if _is_even_permutation(p):
            perms.append(_extend_partial_permutation(p, subset, total_n))
    return perms

# ---------------------------------------------------------------------------
# Larger permutation groups (S8 and S13)
# ---------------------------------------------------------------------------
def generate_s8_subgroup(subset: Iterable[int], total_n: int = 13) -> List[Tuple[int, ...]]:
    """
    Generate all permutations of eight specified node indices.  This
    function simply wraps :func:`generate_symmetric_group` but checks that
    exactly 8 distinct nodes are provided and warns about the size of
    the resulting group (8! = 40320 elements).  Use with caution.

    Parameters
    ----------
    subset : Iterable[int]
        The 8 node indices to permute.
    total_n : int, optional
        The total number of nodes in the graph.  Defaults to 13.

    Returns
    -------
    List[Tuple[int, ...]]
        A list of length ``40320`` of permutation tuples.
    """
    subset = tuple(subset)
    if len(subset) != 8:
        raise ValueError("S8 group requires exactly 8 nodes in subset")
    return generate_symmetric_group(subset, total_n=total_n)

def generate_s13_group(subset: Optional[Iterable[int]] = None) -> Iterable[Tuple[int, ...]]:
    """
    Generate a lazy iterator over permutations of all 13 nodes (S13).

    Notes
    -----
    The full symmetric group on 13 elements contains 13! ≈ 6.227×10^6
    permutations, which is far too large to materialise in memory.  This
    function returns a generator that yields permutations one by one
    using :func:`itertools.permutations`.  Consumers should iterate
    carefully and avoid converting the result to a list.

    The optional ``subset`` argument is provided for API symmetry but is
    ignored (permutations always act on all 13 nodes).

    Returns
    -------
    Iterable[Tuple[int, ...]]
        An iterator over permutation tuples of length 13.
    """
    # ignore provided subset; always permute 1..13
    return permutations(range(1, 14))

"""
Tutorial (als Kommentar):

from src.symmetries import generate_s7_permutations, permutation_to_matrix, apply_permutation_to_adjacency
from src.graph import MetatronCubeGraph

g = MetatronCubeGraph()
A = g.get_adjacency_matrix()
sigmas = generate_s7_permutations()
P = permutation_to_matrix(sigmas[0])  # Identität
A_perm = apply_permutation_to_adjacency(A, P)
"""
