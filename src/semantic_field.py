"""
semantic_field.py
------------------

This module introduces a rudimentary semantic field and diagnostic layer to
extend the post‑symbolic cognition engine beyond pure resonance.  It
implements two classes:

* :class:`SemanticField` – maintains a bank of spectral prototypes and
  classifies incoming resonance patterns by spectral similarity.
* :class:`ResonanceDiagnostics` – computes analytic measures on
  resonance signals such as entropy, spectral centroid and sparsity.

These utilities draw inspiration from the QLOGIC documentation, which
describes a modular architecture composed of oscillator cores, spectral
grammar, semantic fields and entropy analyzers【740178375229903†L68-L96】.
The goal here is not to replicate the full system, but to provide a
foundation for emergent semantic processing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np


class SemanticField:
    """A simple semantic field using spectral prototypes.

    Prototypes represent characteristic frequency signatures for
    semantic categories.  When presented with a new resonance
    spectrum, the field computes cosine similarity to each stored
    prototype and returns the best matches.
    """

    def __init__(self) -> None:
        # mapping from name to prototype vector (1D numpy array)
        self.prototypes: Dict[str, np.ndarray] = {}

    def add_prototype(self, name: str, spectrum: Iterable[float]) -> None:
        """Add a new semantic prototype with the given name and spectrum."""
        self.prototypes[name] = np.array(list(spectrum), dtype=float)

    def classify(self, spectrum: Iterable[float], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Classify a spectrum by similarity to stored prototypes.

        Parameters
        ----------
        spectrum : Iterable[float]
            The input magnitude spectrum.
        top_k : int, optional
            Number of top matches to return.  Defaults to 1.

        Returns
        -------
        List[Tuple[str, float]]
            A list of (name, similarity) pairs sorted by decreasing
            similarity.  Similarity is measured via cosine similarity.
        """
        s = np.array(list(spectrum), dtype=float)
        # avoid division by zero
        if np.linalg.norm(s) == 0:
            return []
        sims = []
        for name, proto in self.prototypes.items():
            if len(proto) != len(s):
                continue
            # cosine similarity
            sim = float(np.dot(s, proto) / (np.linalg.norm(s) * np.linalg.norm(proto) + 1e-12))
            sims.append((name, sim))
        # return top matches
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]


class ResonanceDiagnostics:
    """Compute diagnostic measures on resonance signals.

    Provides static methods for entropy, spectral centroid and sparsity.
    These measures help quantify the structure of resonance spectra and
    can be used for adaptive control and self‑evaluation.
    """

    @staticmethod
    def entropy(spectrum: Iterable[float]) -> float:
        s = np.abs(np.array(list(spectrum), dtype=float))
        s = s / (s.sum() + 1e-12)
        return float(-np.sum(s * np.log2(s + 1e-12)))

    @staticmethod
    def spectral_centroid(spectrum: Iterable[float]) -> float:
        s = np.abs(np.array(list(spectrum), dtype=float))
        if s.sum() == 0:
            return 0.0
        freqs = np.arange(len(s))
        return float(np.sum(freqs * s) / (s.sum() + 1e-12))

    @staticmethod
    def sparsity(spectrum: Iterable[float]) -> float:
        s = np.abs(np.array(list(spectrum), dtype=float))
        if len(s) == 0:
            return 0.0
        return float(np.sum(s > 1e-6) / len(s))