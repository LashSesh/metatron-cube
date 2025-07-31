"""
resonance_tensor.py
--------------------

This module defines a simple resonant tensor field used for simulating
the multidimensional resonance dynamics described in the Metatron‑QDASH
blueprint【248973737581236†L556-L567】.  It serves as a first step towards
modelling tripolar oscillatory fields and tensor dynamics in a
post‑symbolic cognition engine.

The core class :class:`ResonanceTensorField` manages a 3‑dimensional
grid of oscillators.  Each cell in the grid has amplitude ``A``,
frequency ``ω`` and phase ``ϕ`` parameters.  At each time step the
resonance value is updated according to::

    R(t)[i,j,k] = A[i,j,k] * sin(ω[i,j,k] * t + ϕ[i,j,k])

External input can modulate either the amplitudes or the phase offsets.
The class provides methods to step the field forward in time, compute
global coherence metrics and detect singularity events when the
resonance stabilises below a given gradient threshold【248973737581236†L574-L584】.

The implementation is intentionally lightweight and should be viewed
as a prototype to be refined with more realistic tensor dynamics,
entanglement and feedback coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class ResonanceTensorField:
    """A simple 3D resonance tensor field.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Dimensions of the resonance tensor (Nx, Ny, Nz).
    initial_amplitude : float, optional
        Initial amplitude for all cells.  Defaults to 1.0.
    initial_frequency : float, optional
        Initial frequency for all cells.  Defaults to 1.0.
    initial_phase : float, optional
        Initial phase offset (in radians) for all cells.  Defaults to 0.0.
    gradient_threshold : float, optional
        Threshold on the L2 norm of the gradient between successive
        resonance states below which a singularity is considered to
        have occurred.  Defaults to 1e-3.
    """

    shape: Tuple[int, int, int]
    initial_amplitude: float = 1.0
    initial_frequency: float = 1.0
    initial_phase: float = 0.0
    gradient_threshold: float = 1e-3
    time: float = 0.0
    _amplitude: np.ndarray = field(init=False, repr=False)
    _frequency: np.ndarray = field(init=False, repr=False)
    _phase: np.ndarray = field(init=False, repr=False)
    _prev_state: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._amplitude = np.full(self.shape, float(self.initial_amplitude), dtype=float)
        self._frequency = np.full(self.shape, float(self.initial_frequency), dtype=float)
        self._phase = np.full(self.shape, float(self.initial_phase), dtype=float)

    def get_state(self) -> np.ndarray:
        """Return the current resonance values R(t) as a 3D array."""
        t = self.time
        return self._amplitude * np.sin(self._frequency * t + self._phase)

    def step(self, dt: float, input_modulation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance the resonance field by ``dt`` in time.

        Optionally modulate the amplitude and/or phase using an input
        tensor ``input_modulation`` of the same shape.  If provided, the
        modulation is added to the phase parameter, allowing external
        signals to perturb the resonance pattern.

        Parameters
        ----------
        dt : float
            Time increment.
        input_modulation : numpy.ndarray, optional
            A 3D array matching the field shape.  Values are added to
            the phase offsets before computing the new state.

        Returns
        -------
        np.ndarray
            The new resonance state R(t + dt).
        """
        # Update phase with input modulation
        if input_modulation is not None:
            if input_modulation.shape != self.shape:
                raise ValueError("input_modulation must have the same shape as the field")
            self._phase += input_modulation
        # Save previous state for gradient computation
        prev = self.get_state()
        # Update time
        self.time += dt
        new_state = self.get_state()
        self._prev_state = prev
        return new_state

    def coherence(self) -> float:
        """
        Compute a global coherence metric from the current state.

        The coherence is defined as the mean pairwise cosine similarity
        between all cells in the resonance tensor.  It generalises the
        Mandorla resonance calculation to three dimensions.
        """
        R = self.get_state().flatten()
        # normalise cell vectors (each cell is a scalar here, so we
        # consider the distribution of values across the grid)
        if np.allclose(R, 0):
            return 0.0
        # compute similarity between all pairs
        similarities = []
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                a = R[i]
                b = R[j]
                # treat scalars as 1‑D vectors
                sim = (a * b) / ((abs(a) + 1e-12) * (abs(b) + 1e-12))
                similarities.append(sim)
        return float(np.mean(similarities)) if similarities else 0.0

    def gradient_norm(self) -> float:
        """
        Compute the L2 norm of the difference between the current state
        and the previous state.  If no previous state is stored, zero
        is returned.
        """
        if self._prev_state is None:
            return 0.0
        diff = self.get_state() - self._prev_state
        return float(np.linalg.norm(diff))

    def detect_singularity(self) -> bool:
        """
        Determine whether the field has reached a singularity (stabilised).

        A singularity event is considered to occur when the gradient norm
        between successive states falls below ``gradient_threshold``【248973737581236†L574-L584】.
        """
        return self.gradient_norm() < self.gradient_threshold
