
# src/mandorla.py

import numpy as np
from typing import List, Optional

class MandorlaField:
    """
    Globales Entscheidungs- und Resonanzfeld (Vesica Piscis / Mandorla):
    Koppelt Inputs (Seeds, SpiralMemory, GabrielCells, externe Sensorik),
    summiert Feldresonanz und löst bei maximaler Konvergenz Exkalibrationsereignisse aus.
    """

    def __init__(self, threshold: float = 0.985, alpha: float = 0.5, beta: float = 0.5):
        self.inputs: List[np.ndarray] = []
        # Static threshold for backwards compatibility.  If ``alpha`` or
        # ``beta`` are provided, a dynamic threshold θ(t) = α⋅Entropy + β⋅Variance
        # is used instead of this constant value【248973737581236†L520-L533】.
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
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

    def calc_entropy(self) -> float:
        """Compute the entropy of the current input field.

        Each input vector is normalised to yield a probability distribution.
        The Shannon entropy is computed across concatenated inputs.  If no
        inputs are present, zero entropy is returned.
        """
        if not self.inputs:
            return 0.0
        # concatenate all input vectors to a single distribution
        data = np.concatenate([np.abs(v) for v in self.inputs])
        data = data / (data.sum() + 1e-12)
        entropy = -np.sum(data * np.log2(data + 1e-12))
        return float(entropy)

    def calc_variance(self) -> float:
        """Compute the variance of the input amplitudes across all inputs.

        The variance captures how spread out the current resonance field is.
        """
        if not self.inputs:
            return 0.0
        data = np.stack(self.inputs)
        return float(np.var(data))

    def decision_trigger(self) -> bool:
        """Determine whether a decision should be triggered.

        If dynamic threshold parameters ``alpha`` or ``beta`` are non‑zero,
        the threshold is computed at each call as θ(t) = α⋅Entropy + β⋅Variance【248973737581236†L526-L533】.
        Otherwise the static ``threshold`` value is used.  The method
        returns ``True`` if the current resonance exceeds θ(t).  The
        computed threshold is stored in ``self.current_theta`` for
        inspection.
        """
        res = self.calc_resonance()
        # dynamic threshold
        if self.alpha != 0 or self.beta != 0:
            entropy = self.calc_entropy()
            var = self.calc_variance()
            self.current_theta = self.alpha * entropy + self.beta * var
        else:
            self.current_theta = self.threshold
        return res > self.current_theta

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
