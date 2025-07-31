"""
meta_interpreter.py
-------------------

This module implements a minimal meta‑control layer for the post‑symbolic
cognition engine.  The ``MetaInterpreter`` observes the internal
resonance and decision history of a :class:`QDASHAgent` and adjusts
parameters such as the Mandorla dynamic threshold coefficients (α, β)
and oscillator frequencies to maintain balanced decision dynamics.  The
goal is to emulate the self‑modeling and adaptive behaviour described
in the QLOGIC specification【740178375229903†L68-L96】.

The adaptation rules implemented here are simplistic: they increase the
threshold sensitivity when decisions happen too frequently, and
decrease it when the system is overly inert.  Similarly, oscillator
frequencies can be modulated to explore different resonance regimes.
Future versions might incorporate reinforcement learning, entropy
minimisation or more sophisticated homeostasis.
"""

from __future__ import annotations

from typing import Deque
from collections import deque

from .qdash_agent import QDASHAgent


class MetaInterpreter:
    """Adaptive controller for a QDASHAgent.

    Observes decision outcomes and resonance history, adjusting
    ``MandorlaField`` thresholds and oscillator frequencies to maintain
    a target decision rate.  It maintains a sliding window of recent
    decisions to gauge activity.  All adjustments are performed in
    place on the provided agent instance.
    """

    def __init__(self, agent: QDASHAgent, window_size: int = 10, target_rate: float = 0.5) -> None:
        self.agent = agent
        self.window: Deque[bool] = deque(maxlen=window_size)
        self.target_rate = target_rate  # desired ratio of true decisions

    def record_decision(self) -> None:
        """Record the most recent decision from the agent."""
        self.window.append(bool(self.agent.last_decision))

    def adjust_parameters(self) -> None:
        """Adjust Mandorla threshold coefficients based on recent decision rate.

        If the agent makes decisions too frequently (above
        ``target_rate``), increase the threshold by scaling α and β up.
        If decisions are too rare, decrease the threshold.  The
        adjustments are small (±5 %) to allow gradual adaptation.
        """
        if not self.window:
            return
        rate = sum(self.window) / len(self.window)
        # determine adjustment factor
        if rate > self.target_rate + 0.1:
            # decisions too frequent -> raise threshold
            self.agent.mandorla.alpha *= 1.05
            self.agent.mandorla.beta *= 1.05
        elif rate < self.target_rate - 0.1:
            # decisions too rare -> lower threshold
            self.agent.mandorla.alpha *= 0.95
            self.agent.mandorla.beta *= 0.95

    def modulate_oscillator_frequency(self, factor: float) -> None:
        """Scale oscillator frequencies by ``factor``.

        This can be used to shift the resonance spectrum into different
        regimes.  Only the oscillator core frequencies are modified.
        """
        oc = self.agent.qlogic.osc_core
        # In this simple oscillator, we adjust the base frequency by
        # scaling the angular increment of the sinusoid: changing the
        # spacing of phases corresponds to frequency modulation.
        # We implement this by scaling the number of nodes, which is a
        # hackish but effective way to modify the period of the sine.
        # Users can subclass QLOGICOscillatorCore for finer control.
        oc.num_nodes = max(1, int(round(oc.num_nodes * factor)))