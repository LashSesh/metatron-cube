
# src/mandorla.py

import numpy as np
from typing import List, Optional

class MandorlaField:
    """
    Globales Entscheidungs- und Resonanzfeld (Vesica Piscis / Mandorla):
    Koppelt Inputs (Seeds, SpiralMemory, GabrielCells, externe Sensorik),
    summiert Feldresonanz und löst bei maximaler Konvergenz Exkalibrationsereignisse aus.
    """

    def __init__(self, threshold: float = 0.985):
        self.inputs: List[np.ndarray] = []
        self.threshold = threshold
        self.resonance = 0.0
        self.history = []

    def add_input(self, vec: np.ndarray):
        self.inputs.append(vec)

    def clear_inputs(self):
        self.inputs = []

    def calc_resonance(self) -> float:
        if len(self.inputs) < 2:
            self.resonance = 0.0
            return 0.0
        res = []
        for i in range(len(self.inputs)):
            for j in range(i+1, len(self.inputs)):
                a = self.inputs[i]
                b = self.inputs[j]
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                res.append(sim)
        self.resonance = float(np.mean(res))
        self.history.append(self.resonance)
        return self.resonance

    def decision_trigger(self) -> bool:
        return self.calc_resonance() > self.threshold

"""
Beispiel/Tutorial:
from src.mandorla import MandorlaField
import numpy as np

mf = MandorlaField()
v1 = np.random.rand(5)
v2 = np.random.rand(5)
mf.add_input(v1)
mf.add_input(v2)
print("Resonanz:", mf.calc_resonance())
if mf.decision_trigger():
    print("Exkalibration: Entscheidung ausgelöst!")
"""
