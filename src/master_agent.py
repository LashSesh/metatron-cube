
# src/master_agent.py

"""Master agent combining SpiralMemory, GabrielCells, MandorlaField and SeraphicFeedback."""

from .spiralmemory import SpiralMemory
from .gabriel_cell import GabrielCell
from .mandorla import MandorlaField
from .seraphic_feedback import SeraphicFeedbackModule

import numpy as np

class MasterAgent:
    """
    Vereint SpiralMemory, GabrielCells, MandorlaField, SeraphicFeedback und kann vollständigen
    Blueprint-gemäßen Resonanz-Workflow ausführen (Input bis Decision).
    """

    def __init__(self, n_cells: int = 4, mandorla_alpha: float = 0.5, mandorla_beta: float = 0.5):
        self.spiral_memory = SpiralMemory(alpha=0.07)
        self.gabriel_cells = [GabrielCell() for _ in range(n_cells)]
        for i in range(n_cells-1):
            self.gabriel_cells[i].couple(self.gabriel_cells[i+1])
        # initialise MandorlaField with dynamic threshold parameters
        self.mandorla = MandorlaField(alpha=mandorla_alpha, beta=mandorla_beta)
        self.seraphic = SeraphicFeedbackModule()
        self.last_decision = None

    def process(self, raw_inputs):
        vecs = self.seraphic.map_inputs(raw_inputs)
        points, psi = self.spiral_memory.step([str(i) for i in raw_inputs], max_iter=18)
        for cell, vec in zip(self.gabriel_cells, points[:len(self.gabriel_cells)]):
            cell.activate(inp=np.sum(vec))
        self.mandorla.clear_inputs()
        for cell in self.gabriel_cells:
            self.mandorla.add_input(np.ones(5) * cell.output)
        decision = self.mandorla.decision_trigger()
        self.last_decision = decision
        return {
            "inputs": raw_inputs,
            "spiral_points": points,
            "gabriel_outputs": [c.output for c in self.gabriel_cells],
            "mandorla_resonance": self.mandorla.resonance,
            "decision": decision,
        }
