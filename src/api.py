
# src/api.py

import json
import numpy as np
from typing import Iterable, Tuple
from .graph import MetatronCubeGraph
from .symmetries import permutation_matrix

def export_nodes_json(graph: MetatronCubeGraph, path: str = None) -> str:
    nodes = [
        {
            "index": node.index,
            "label": node.label,
            "coords": node.coords
        }
        for node in graph.nodes
    ]
    js = json.dumps(nodes, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(js)
    return js

def export_edges_json(graph: MetatronCubeGraph, path: str = None) -> str:
    edges = [
        {"source": i, "target": j}
        for i, j in graph.edges
    ]
    js = json.dumps(edges, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(js)
    return js

def export_adjacency_json(graph: MetatronCubeGraph, path: str = None) -> str:
    matrix = graph.get_adjacency_matrix().tolist()
    js = json.dumps(matrix, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(js)
    return js


def export_group_json(perms: Iterable[Tuple[int, ...]], path: str = None) -> str:
    """Export a list of permutation tuples as JSON.

    Parameters
    ----------
    perms : Iterable[Tuple[int, ...]]
        An iterable of permutation tuples (1‑based indices).
    path : str, optional
        If provided, the JSON string is written to this file.

    Returns
    -------
    str
        A JSON string representing the permutations as lists of integers.
    """
    data = [list(p) for p in perms]
    js = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(js)
    return js


def export_matrices_json(perms: Iterable[Tuple[int, ...]], size: int = 13, path: str = None) -> str:
    """Export permutation matrices to JSON.

    Each permutation is converted into a ``size``×``size`` matrix using
    :func:`symmetries.permutation_matrix`.  The resulting 3D array (list
    of matrices) is encoded as JSON.

    Parameters
    ----------
    perms : Iterable[Tuple[int, ...]]
        Permutations to convert to matrices.
    size : int, optional
        Dimension of the matrices.  Defaults to 13.
    path : str, optional
        If provided, write the JSON string to this file.

    Returns
    -------
    str
        JSON string containing the list of matrices.
    """
    mats = [permutation_matrix(p, size).tolist() for p in perms]
    js = json.dumps(mats, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(js)
    return js

"""
Verwendung:
from src.graph import MetatronCubeGraph
from src.api import export_nodes_json, export_edges_json, export_adjacency_json

g = MetatronCubeGraph()
print(export_nodes_json(g))
print(export_edges_json(g))
print(export_adjacency_json(g))
"""
