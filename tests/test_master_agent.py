
# tests/test_master_agent.py

from src.master_agent import MasterAgent

def test_master_agent_process():
    agent = MasterAgent(n_cells=3)
    result = agent.process(['CYBER', 7.5, [1,2,3]])
    assert "spiral_points" in result
    assert "gabriel_outputs" in result
    assert "mandorla_resonance" in result
    assert "decision" in result
