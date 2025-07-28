
# src/api.py

import json
import numpy as np
from .graph import MetatronCubeGraph

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

"""
Verwendung:
from src.graph import MetatronCubeGraph
from src.api import export_nodes_json, export_edges_json, export_adjacency_json

g = MetatronCubeGraph()
print(export_nodes_json(g))
print(export_edges_json(g))
print(export_adjacency_json(g))
"""
