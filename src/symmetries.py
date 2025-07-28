
# src/symmetries.py

import numpy as np
from itertools import permutations
from typing import List, Tuple, Callable

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
