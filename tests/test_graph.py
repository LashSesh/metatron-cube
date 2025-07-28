
# tests/test_graph.py

import numpy as np
from src.graph import MetatronCubeGraph

def test_adjacency_matrix_shape_and_symmetry():
    g = MetatronCubeGraph()
    A = g.get_adjacency_matrix()
    assert A.shape == (13, 13)
    assert np.array_equal(A, A.T), "Adjazenzmatrix muss symmetrisch sein."

def test_neighbors_of_center():
    g = MetatronCubeGraph()
    neighbors = g.get_neighbors(1)  # 1 = Center
    labels = sorted([n.label for n in neighbors])
    assert len(neighbors) == 6
    assert labels == [f"H{i}" for i in range(1, 7)]

def test_edge_list_indices_valid():
    g = MetatronCubeGraph()
    indices = {node.index for node in g.nodes}
    for i, j in g.get_edge_list():
        assert i in indices
        assert j in indices

def test_no_duplicate_edges():
    g = MetatronCubeGraph()
    edge_set = set()
    for i, j in g.get_edge_list():
        edge = tuple(sorted((i, j)))
        assert edge not in edge_set, "Doppelte Kante gefunden"
        edge_set.add(edge)
