
# src/qlogic.py

import numpy as np

class QLOGICOscillatorCore:
    """Simuliert ein einfaches Oszillatornetz – erzeugt Resonanzmuster als Feldvektor."""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def generate_pattern(self, t: float = 0.0) -> np.ndarray:
        phases = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=False)
        return np.sin(phases + t)

class SpectralGrammar:
    """Wandelt Feldvektor in 'Bedeutungsfrequenzen' um."""
    def analyze(self, field: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.fft(field))

class EntropyAnalyzer:
    """Bewertet die Kohärenz (Ordnung) des aktuellen Feldes."""
    def entropy(self, field: np.ndarray) -> float:
        p = np.abs(field)
        p = p / (p.sum() + 1e-12)
        return -np.sum(p * np.log2(p + 1e-12))

class QLogicEngine:
    """Hauptengine, führt Resonanzberechnung und Self-Model durch."""
    def __init__(self, num_nodes: int):
        self.osc_core = QLOGICOscillatorCore(num_nodes)
        self.grammar = SpectralGrammar()
        self.analyzer = EntropyAnalyzer()

    def step(self, t: float = 0.0):
        field = self.osc_core.generate_pattern(t)
        freq = self.grammar.analyze(field)
        entropy = self.analyzer.entropy(field)
        return {"field": field, "spectrum": freq, "entropy": entropy}
