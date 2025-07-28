
# tests/test_geometry.py

import pytest
from src.geometry import get_metatron_nodes, get_metatron_edges
import numpy as np

def test_node_count():
    nodes = get_metatron_nodes()
    assert len(nodes) == 13, "Es müssen 13 Knoten vorhanden sein."

def test_center_node():
    nodes = get_metatron_nodes()
    center = nodes[0]
    assert center.label == "C"
    assert center.coords == (0.0, 0.0, 0.0)

def test_hexagon_nodes_are_on_unit_circle():
    nodes = get_metatron_nodes()
    for node in nodes[1:7]:
        x, y, z = node.coords
        assert np.isclose(z, 0.0)
        r = np.sqrt(x**2 + y**2)
        assert np.isclose(r, 1.0, atol=1e-8), f"{node.label} ist nicht auf Einheitskreis"

def test_cube_nodes_have_correct_coords():
    nodes = get_metatron_nodes()
    sqrt2_inv = 1/np.sqrt(2)
    cube_coords = set()
    for node in nodes[7:]:
        for v in node.coords:
            assert np.isclose(abs(v), sqrt2_inv), f"{node.label} hat falsche Koordinate"
        cube_coords.add(node.coords)
    # Keine Duplikate
    assert len(cube_coords) == 6

def test_edges_reference_valid_nodes():
    nodes = get_metatron_nodes()
    edges = get_metatron_edges()
    valid_indices = {node.index for node in nodes}
    for i, j in edges:
        assert i in valid_indices
        assert j in valid_indices
