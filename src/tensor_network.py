"""
tensor_network.py
------------------

This module introduces a high‑level tensor network container for
managing multiple resonance fields.  While it does not implement
full‑fledged matrix‑product states or tensor contraction algorithms, it
provides an extensible structure for combining several
:class:`ResonanceTensorField` instances into a network and computing
coherence and correlation metrics across them.

By grouping resonance fields together, the network can simulate
entangled interactions between different components of the
post‑symbolic engine.  For example, two separate 3D resonance grids
could represent different conceptual domains whose couplings lead to
emergent behaviour.  The network measures both intra‑field coherence
and inter‑field cross‑coherence, offering a proxy for entanglement
without requiring an exact quantum description.
"""

from __future__ import annotations

from typing import List, Optional, Iterable
import numpy as np

from .resonance_tensor import ResonanceTensorField


class TensorNetwork:
    """A simple container for multiple resonance tensor fields.

    The network manages an arbitrary number of :class:`ResonanceTensorField`
    instances, stepping them together and computing joint metrics.

    Parameters
    ----------
    fields : Iterable[ResonanceTensorField], optional
        Initial set of resonance fields to include in the network.
    """

    def __init__(self, fields: Optional[Iterable[ResonanceTensorField]] = None) -> None:
        self.fields: List[ResonanceTensorField] = list(fields) if fields is not None else []

    def add_field(self, field: ResonanceTensorField) -> None:
        """Add a new field to the network."""
        self.fields.append(field)

    def step(self, dt: float, input_modulations: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Step all fields forward in time.

        Parameters
        ----------
        dt : float
            Time increment for all fields.
        input_modulations : list of numpy.ndarray, optional
            If provided, must match the number of fields; each element
            is passed as ``input_modulation`` to the corresponding field.

        Returns
        -------
        List[np.ndarray]
            A list of the new states of each field.
        """
        new_states = []
        for idx, field in enumerate(self.fields):
            mod = None
            if input_modulations is not None and idx < len(input_modulations):
                mod = input_modulations[idx]
            new_states.append(field.step(dt, input_modulation=mod))
        return new_states

    def coherence(self) -> float:
        """Compute the mean coherence across all fields.

        Returns the average of the individual coherence values."""
        if not self.fields:
            return 0.0
        return float(np.mean([f.coherence() for f in self.fields]))

    def cross_coherence(self) -> float:
        """
        Compute the cross‑coherence between all pairs of fields.

        Each field state is flattened to a 1D vector; the mean cosine
        similarity between all pairs of fields is returned.  This
        provides a simple measure of correlation/entanglement between
        different resonance grids.
        """
        if len(self.fields) < 2:
            return 0.0
        vectors = [f.get_state().flatten() for f in self.fields]
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a = vectors[i]
                b = vectors[j]
                num = np.dot(a, b)
                denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                sims.append(num / denom)
        return float(np.mean(sims))

    def detect_singularities(self) -> bool:
        """Return True if any field in the network has stabilised."""
        return any(field.detect_singularity() for field in self.fields)