
# tests/test_qlogic.py

from src.qlogic import QLogicEngine

def test_qlogic_step():
    engine = QLogicEngine(num_nodes=13)
    res = engine.step(0.0)
    assert "field" in res and "spectrum" in res and "entropy" in res
    assert len(res["field"]) == 13
