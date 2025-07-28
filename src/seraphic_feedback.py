
# src/seraphic_feedback.py

import numpy as np
from typing import Any, List, Callable

class SeraphicFeedbackModule:
    """
    Adapter für externe Inputs (Strings, Zahlen, Sensoren, API), 
    übersetzt zu 5D-Resonanzimpulsen für SpiralMemory oder MandorlaField.
    Unterstützt einfache Filter (z.B. Normalisierung, Transformation, Mapping).
    """

    def __init__(self, filter_func: Callable[[Any], np.ndarray] = None):
        if filter_func is None:
            self.filter_func = self.default_filter
        else:
            self.filter_func = filter_func

    def default_filter(self, x: Any) -> np.ndarray:
        if isinstance(x, str):
            base = np.array([ord(c) for c in x])
            v = np.zeros(5)
            for i in range(5):
                v[i] = np.sum(base * np.cos(2 * np.pi * (i+1) * np.arange(len(base)) / (len(base) + 5)))
            norm = np.linalg.norm(v)
            return v / (norm + 1e-12)
        elif isinstance(x, (int, float)):
            v = np.ones(5) * float(x)
            norm = np.linalg.norm(v)
            return v / (norm + 1e-12)
        elif isinstance(x, (list, tuple, np.ndarray)):
            vals = np.array(x)
            mean = np.mean(vals)
            v = np.ones(5) * mean
            norm = np.linalg.norm(v)
            return v / (norm + 1e-12)
        else:
            raise ValueError("Unsupported input type for SeraphicFeedbackModule.")

    def map_inputs(self, inputs: List[Any]) -> List[np.ndarray]:
        return [self.filter_func(x) for x in inputs]

"""
Beispiel/Tutorial:
from src.seraphic_feedback import SeraphicFeedbackModule

sfm = SeraphicFeedbackModule()
vecs = sfm.map_inputs(['QLOGIC', 3.14, [1, 2, 3, 4]])
print("Feedback-Vektoren:", vecs)
"""
