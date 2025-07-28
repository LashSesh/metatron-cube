
# tests/test_gabriel_cell.py

from src.gabriel_cell import GabrielCell

def test_activation_and_feedback():
    cell = GabrielCell()
    out1 = cell.activate(inp=0.8)
    cell.feedback(target=1.5)
    out2 = cell.activate()
    assert abs(out2 - out1) > 0

def test_coupling_and_neighbor_feedback():
    c1 = GabrielCell(psi=0.4)
    c2 = GabrielCell(psi=0.9)
    c1.couple(c2)
    c1.activate(inp=0.5)
    c2.activate(inp=1.2)
    c1.neighbor_feedback()
    c2.neighbor_feedback()
    assert c1.psi != 0.4 and c2.psi != 0.9
