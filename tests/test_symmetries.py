
# tests/test_symmetries.py

import numpy as np
from src.symmetries import (
    generate_s7_permutations, permutation_to_matrix, apply_permutation_to_adjacency
)
from src.graph import MetatronCubeGraph

def test_number_of_s7_permutations():
    sigmas = generate_s7_permutations()
    assert len(sigmas) == 5040

def test_permutation_matrix_properties():
    sigmas = generate_s7_permutations()
    for sigma in sigmas[:5]:  # Teste nur die ersten 5, Performance!
        P = permutation_to_matrix(sigma)
        # Jede Zeile/Splate hat genau eine 1
        assert np.all(P.sum(axis=0) == 1)
        assert np.all(P.sum(axis=1) == 1)
        # Nur 0 oder 1
        assert set(np.unique(P)) <= {0, 1}

def test_identity_permutation_preserves_adjacency():
    g = MetatronCubeGraph()
    A = g.get_adjacency_matrix()
    sigma_id = tuple(range(1, 8))  # Identität
    P = permutation_to_matrix(sigma_id)
    A_new = apply_permutation_to_adjacency(A, P)
    assert np.array_equal(A, A_new)

def test_permute_and_inverse_restores_adjacency():
    g = MetatronCubeGraph()
    A = g.get_adjacency_matrix()
    sigmas = generate_s7_permutations()
    sigma = sigmas[42]  # Beispiel
    # Berechne Inverse (Index → Position in sigma)
    sigma_inv = tuple(sorted(range(1, 8), key=lambda i: sigma.index(i) + 1))
    P = permutation_to_matrix(sigma)
    P_inv = permutation_to_matrix(sigma_inv)
    A_perm = apply_permutation_to_adjacency(A, P)
    A_restored = apply_permutation_to_adjacency(A_perm, P_inv)
    assert np.array_equal(A, A_restored)
