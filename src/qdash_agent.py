"""
qdash_agent.py
---------------

This module implements a simplified QDASH (Quantum‑Dash) agent by
combining the oscillator logic from QLOGIC with the Mandorla resonance
field and SpiralMemory.  The agent follows the decision cycle
outlined in the Metatron‑QDASH blueprint【248973737581236†L480-L509】:

1. **Input transduction** – External input is transformed into an
   oscillator signal using the QLOGIC oscillator core.  In this
   minimal implementation we modulate the oscillator amplitudes by
   aggregating the input vector.
2. **Resonance coupling** – The oscillator signal is added to the
   Mandorla field along with the current Gabriel cell outputs (if
   provided) to compute a resonance pattern.
3. **Coherence measurement** – The global resonance value (mean cosine
   similarity) acts as coherence metric.
4. **Singularity trigger** – If coherence exceeds the adaptive
   threshold θ(t)=α⋅Entropy+β⋅Variance【248973737581236†L526-L533】, a decision
   event is emitted.
5. **Feedback encoding** – The internal state (SpiralMemory and
   Gabriel cells) is updated with the new input.

This implementation is intentionally high‑level; it does not yet
include full tensor dynamics, entanglement or Ophan kernels.  It is
intended as a stepping stone toward the complete QDASH architecture.
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict, Any
import numpy as np

from .qlogic import QLogicEngine
from .mandorla import MandorlaField
from .spiralmemory import SpiralMemory
from .gabriel_cell import GabrielCell


class QDASHAgent:
    """A resonant agent implementing a simplified QDASH decision cycle.

    Parameters
    ----------
    n_cells : int, optional
        Number of Gabriel cells used for feedback coupling.  Defaults to 4.
    alpha : float, optional
        Weight of entropy term in the adaptive threshold.  Defaults to 0.5.
    beta : float, optional
        Weight of variance term in the adaptive threshold.  Defaults to 0.5.
    """

    def __init__(self, n_cells: int = 4, alpha: float = 0.5, beta: float = 0.5) -> None:
        # set up QLogic engine with optional semantic field; prototypes can be
        # registered externally on qlogic.semantic_field
        try:
            from .semantic_field import SemanticField
            sf = SemanticField()
        except Exception:
            sf = None
        self.qlogic = QLogicEngine(num_nodes=13, semantic_field=sf)  # 13 nodes for oscillator pattern
        self.mandorla = MandorlaField(alpha=alpha, beta=beta)
        self.spiral = SpiralMemory(alpha=0.07)
        self.cells = [GabrielCell() for _ in range(n_cells)]
        for i in range(n_cells - 1):
            self.cells[i].couple(self.cells[i + 1])
        # maintain time parameter for oscillator phase
        self.time = 0.0
        # last decision output
        self.last_decision: Optional[bool] = None

    def trm_transform(self, input_vector: Iterable[float]) -> np.ndarray:
        """Transform external input into an oscillator signal.

        The QDASH blueprint references a transformation TRM.Transform that
        maps sensory input to oscillator signatures.  In this minimal
        implementation we sum the input and scale the QLOGIC oscillator
        pattern accordingly.  Future work should implement a true
        tripolar resonant map.
        """
        inp = np.array(list(input_vector), dtype=float)
        amplitude = float(inp.sum())  # naive aggregation
        osc_pattern = self.qlogic.osc_core.generate_pattern(self.time)
        return amplitude * osc_pattern

    def update_internal_state(self, spiral_points: Iterable[np.ndarray]) -> None:
        """Update Gabriel cells from new SpiralMemory points and push to Mandorla."""
        for cell, vec in zip(self.cells, spiral_points):
            cell.activate(inp=np.sum(vec))
        # add Gabriel outputs as resonance contributions
        for cell in self.cells:
            self.mandorla.add_input(np.ones(5) * cell.output)

    def decision_cycle(self, input_vector: Iterable[float], max_iter: int = 3, dt: float = 1.0) -> Dict[str, Any]:
        """Run the QDASH decision cycle on a single input vector.

        Parameters
        ----------
        input_vector : Iterable[float]
            External input to be processed.
        max_iter : int, optional
            Maximum number of resonance iterations before returning.  If
            coherence never exceeds the threshold, the method returns
            after ``max_iter`` cycles.  Defaults to 3.
        dt : float, optional
            Time increment between iterations (controls oscillator phase).
            Defaults to 1.0.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the oscillator signal, resonance,
            adaptive threshold, decision flag and internal states.
        """
        # Reset Mandorla inputs
        self.mandorla.clear_inputs()
        # Step the SpiralMemory to embed the input (use string representation)
        points, psi = self.spiral.step([str(x) for x in input_vector], max_iter=18)
        # Update Gabriel cells and add outputs to Mandorla
        self.update_internal_state(points[:len(self.cells)])
        # Generate oscillator signal from input
        osc_signal = self.trm_transform(input_vector)
        # Add oscillator pattern to Mandorla inputs
        self.mandorla.add_input(osc_signal)
        # iterate resonance until decision or max_iter
        decision = False
        for _ in range(max_iter):
            c = self.mandorla.calc_resonance()
            threshold = self.mandorla.current_theta if hasattr(self.mandorla, 'current_theta') else self.mandorla.threshold
            if c > threshold:
                decision = True
                break
            # wait and update time for oscillator phase
            self.time += dt
            # update oscillator signal for next iteration and add as input
            osc_signal = self.trm_transform(input_vector)
            self.mandorla.add_input(osc_signal)
        self.last_decision = decision
        return {
            "oscillator_signal": osc_signal.tolist(),
            "resonance": float(self.mandorla.resonance),
            "threshold": float(self.mandorla.current_theta if hasattr(self.mandorla, 'current_theta') else self.mandorla.threshold),
            "decision": decision,
            "spiral_points": [p.tolist() for p in points],
            "gabriel_outputs": [c.output for c in self.cells],
        }