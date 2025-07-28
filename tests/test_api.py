
# tests/test_api.py

from src.graph import MetatronCubeGraph
from src.api import export_nodes_json, export_edges_json, export_adjacency_json
import json

def test_export_nodes_json():
    g = MetatronCubeGraph()
    js = export_nodes_json(g)
    data = json.loads(js)
    assert isinstance(data, list)
    assert len(data) == 13
    assert "coords" in data[0]

def test_export_edges_json():
    g = MetatronCubeGraph()
    js = export_edges_json(g)
    data = json.loads(js)
    assert isinstance(data, list)
    assert "source" in data[0] and "target" in data[0]

def test_export_adjacency_json():
    g = MetatronCubeGraph()
    js = export_adjacency_json(g)
    data = json.loads(js)
    assert isinstance(data, list)
    assert len(data) == 13
    assert len(data[0]) == 13
