
# src/agent.py

from .graph import MetatronCubeGraph
from .qlogic import QLogicEngine
from .monolith import TripolarOperator, OphanKernel, MonolithDecision
import numpy as np

class MetatronAgent:
    """Fusioniert Cube, QLogic und Monolith – agiert als Loop-Agent."""
    def __init__(self):
        self.graph = MetatronCubeGraph()
        self.qlogic = QLogicEngine(num_nodes=len(self.graph.nodes))
        self.monolith = MonolithDecision(
            OphanKernel([
                TripolarOperator(np.random.rand(), np.random.rand(), np.random.rand())
                for _ in range(4)
            ])
        )
        self.memory = []

    def step(self, t: float = 0.0):
        qres = self.qlogic.step(t)
        state = self.monolith.step()
        self.memory.append({"t": t, "entropy": qres["entropy"], "decision": state})
        return {"qlogic": qres, "monolith": state}
