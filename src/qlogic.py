
# src/qlogic.py

import numpy as np
from typing import Optional, Dict, Any

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
    """Hauptengine für spektrale Verarbeitung und semantische Analyse.

    Diese Engine erzeugt über den ``QLOGICOscillatorCore`` Schwingungsmuster,
    wendet die ``SpectralGrammar`` an, berechnet Entropie und kann über
    einen optionalen ``SemanticField`` semantische Klassifikationen
    ausgeben.  Darüber hinaus steht eine ``ResonanceDiagnostics`` zur
    Verfügung, um weitere Kennwerte (z. B. Spektralzentroid, Sparsity) zu
    liefern.
    """
    def __init__(self, num_nodes: int, semantic_field: Optional[object] = None) -> None:
        self.osc_core = QLOGICOscillatorCore(num_nodes)
        self.grammar = SpectralGrammar()
        self.analyzer = EntropyAnalyzer()
        # optional semantic field for classification
        self.semantic_field = semantic_field
        # import diagnostics lazily to avoid circular deps
        try:
            from .semantic_field import ResonanceDiagnostics
            self.diagnostics = ResonanceDiagnostics
        except Exception:
            self.diagnostics = None

    def step(self, t: float = 0.0) -> Dict[str, any]:
        """Generate an oscillator pattern and analyse it.

        Returns a dictionary with keys ``field``, ``spectrum``, ``entropy``.
        If a semantic field is present, the best match and similarity are
        included under ``classification``.  Additional diagnostic metrics
        may be provided via ``diagnostics``.
        """
        field = self.osc_core.generate_pattern(t)
        spectrum = self.grammar.analyze(field)
        ent = self.analyzer.entropy(field)
        result = {"field": field, "spectrum": spectrum, "entropy": ent}
        # classify via semantic field if available
        if self.semantic_field is not None:
            matches = self.semantic_field.classify(spectrum, top_k=1)
            result["classification"] = matches[0] if matches else None
        # compute diagnostics
        if self.diagnostics is not None:
            try:
                centroid = self.diagnostics.spectral_centroid(spectrum)
                sparsity = self.diagnostics.sparsity(spectrum)
                result["diagnostics"] = {
                    "spectral_centroid": centroid,
                    "sparsity": sparsity,
                }
            except Exception:
                pass
        return result
