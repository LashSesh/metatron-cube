
# tests/test_agent.py

from src.agent import MetatronAgent

def test_agent_step():
    agent = MetatronAgent()
    res = agent.step(0.0)
    assert "qlogic" in res and "monolith" in res
    assert isinstance(agent.memory, list)
