"""
quantum.py
-----------

This module provides basic classes and utilities for representing
quantum‑mechanical states and operators on the Metatron Cube.  A
``QuantumState`` is a 13‑dimensional complex vector corresponding to the
amplitudes of being at each of the 13 canonical nodes.  A
``QuantumOperator`` is a 13×13 matrix (typically unitary) that acts on
these states.  Together they enable a rudimentary Hilbert‑space
formalism for post‑symbolic cognition as envisioned in the Theory of
Everything document【233813512775479†L1590-L1644】.

The initial implementation focuses on basic superposition, inner
products, and permutation‑based unitaries derived from the symmetry
groups of the cube.  Future extensions might include entanglement
across multiple cubes, higher‑order tensor representations, and
non‑permutation gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional
import numpy as np

from .symmetries import permutation_matrix


@dataclass
class QuantumState:
    """A quantum state on the 13‑dimensional Hilbert space of the cube.

    The state is represented internally as a NumPy array of complex
    amplitudes (column vector).  Upon initialization, the state is
    normalised to unit length.  Basic operations such as applying
    operators and computing inner products are provided.

    Parameters
    ----------
    amplitudes : Iterable[complex]
        A sequence of 13 complex numbers representing the amplitudes for
        nodes 1–13.  If fewer than 13 entries are provided, the vector
        will be padded with zeros; if more entries are provided, a
        ``ValueError`` is raised.
    normalize : bool, optional
        If ``True`` (default), the state vector is normalised to have
        Euclidean norm 1.  If ``False``, no normalisation is performed.
    """

    amplitudes: np.ndarray

    def __init__(self, amplitudes: Iterable[complex], normalize: bool = True) -> None:
        amps = list(amplitudes)
        if len(amps) > 13:
            raise ValueError("QuantumState expects at most 13 amplitudes")
        # pad with zeros if necessary
        while len(amps) < 13:
            amps.append(0j)
        self.amplitudes = np.array(amps, dtype=np.complex128)
        if normalize:
            self.normalise()

    def normalise(self) -> 'QuantumState':
        """Normalise the state to unit norm (L2)."""
        nrm = np.linalg.norm(self.amplitudes)
        if nrm == 0:
            # avoid division by zero: define |0⟩
            self.amplitudes = np.zeros_like(self.amplitudes)
        else:
            self.amplitudes = self.amplitudes / nrm
        return self

    def inner_product(self, other: 'QuantumState') -> complex:
        """Return the inner product ⟨ψ|ϕ⟩ between this state and ``other``.

        The inner product is conjugate linear in the first argument and
        linear in the second.  The result is a complex number.
        """
        return np.vdot(self.amplitudes, other.amplitudes)

    def apply(self, operator: 'QuantumOperator') -> 'QuantumState':
        """Apply a quantum operator to this state and return the new state."""
        if operator.matrix.shape != (13, 13):
            raise ValueError("Operator must be 13×13 to act on a QuantumState")
        new_amplitudes = operator.matrix @ self.amplitudes
        return QuantumState(new_amplitudes)

    def probabilities(self) -> np.ndarray:
        """Return the probability distribution |ψ_i|² over the 13 nodes."""
        return np.abs(self.amplitudes) ** 2

    def measure(self) -> int:
        """Perform a projective measurement in the computational basis.

        Returns
        -------
        int
            The index (1‑based) of the measured node.  Measurement
            collapses the state; subsequent calls will collapse relative
            to the post‑measurement state.
        """
        probs = self.probabilities()
        idx = np.random.choice(13, p=probs)
        # collapse to the basis state |idx⟩
        collapsed = np.zeros_like(self.amplitudes)
        collapsed[idx] = 1.0
        self.amplitudes = collapsed
        return idx + 1

    def as_array(self) -> np.ndarray:
        """Return the underlying amplitude vector."""
        return self.amplitudes.copy()

    def __repr__(self) -> str:
        amps = ', '.join(f'{a:.3g}' for a in self.amplitudes)
        return f"QuantumState([{amps}])"


@dataclass
class QuantumOperator:
    """A linear operator acting on the 13‑dimensional state space.

    The operator is represented as a 13×13 complex matrix.  For
    permutation operators, the matrix is unitary (binary entries), but
    the class can hold arbitrary linear operators.  Composition and
    unitarity checks are provided.

    Parameters
    ----------
    matrix : np.ndarray
        A 13×13 NumPy array of complex numbers.
    """

    matrix: np.ndarray

    def __init__(self, matrix: np.ndarray) -> None:
        if matrix.shape != (13, 13):
            raise ValueError("QuantumOperator matrix must be 13×13")
        # ensure dtype is complex
        self.matrix = np.array(matrix, dtype=np.complex128)

    @classmethod
    def from_permutation(cls, sigma: Tuple[int, ...]) -> 'QuantumOperator':
        """Construct a permutation operator from a 13‑length permutation tuple.

        Parameters
        ----------
        sigma : Tuple[int, ...]
            A permutation of (1..13) describing how basis vectors map to
            new positions.  This is typically produced by
            :func:`metatron_cube.src.symmetries.generate_s7_permutations` or
            the high‑level :class:`MetatronCube` group enumeration.

        Returns
        -------
        QuantumOperator
            The corresponding permutation operator as a QuantumOperator.
        """
        P = permutation_matrix(sigma, size=13).astype(np.complex128)
        return cls(P)

    def compose(self, other: 'QuantumOperator') -> 'QuantumOperator':
        """Return the composition (matrix multiplication) of this operator with ``other``."""
        return QuantumOperator(self.matrix @ other.matrix)

    def is_unitary(self, atol: float = 1e-8) -> bool:
        """Check whether the operator is unitary (O⋅O† = I)."""
        I = np.eye(13, dtype=np.complex128)
        return np.allclose(self.matrix @ self.matrix.conj().T, I, atol=atol) and \
               np.allclose(self.matrix.conj().T @ self.matrix, I, atol=atol)

    def __repr__(self) -> str:
        return f"QuantumOperator(matrix=\n{self.matrix})"