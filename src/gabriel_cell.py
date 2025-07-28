
# src/gabriel_cell.py

import numpy as np
from typing import List, Optional

class GabrielCell:
    """
    Minimalistische Feedback-Zelle: modulierbarer Resonator mit (psi, rho, omega).
    Kann auf Feedback und Eingaben dynamisch reagieren, Hebbian Learning implementieren
    und mit Nachbarn koppeln.
    """

    def __init__(self, psi: float = 1.0, rho: float = 1.0, omega: float = 1.0, learn_rate: float = 0.12):
        self.psi = psi  # Aktivierungslevel
        self.rho = rho  # Kohärenz
        self.omega = omega  # Rhythmus/Oszillation
        self.learn_rate = learn_rate
        self.output = self.psi * self.rho * self.omega
        self.neighbors: List['GabrielCell'] = []

    def activate(self, inp: Optional[float] = None) -> float:
        if inp is not None:
            self.psi = (1 - self.learn_rate) * self.psi + self.learn_rate * inp
        self.output = self.psi * self.rho * self.omega
        return self.output

    def feedback(self, target: float):
        err = target - self.output
        self.psi += self.learn_rate * err
        self.rho += self.learn_rate * np.tanh(err)
        self.omega += self.learn_rate * np.sin(err)
        self.psi = np.clip(self.psi, 0.01, 10)
        self.rho = np.clip(self.rho, 0.01, 10)
        self.omega = np.clip(self.omega, 0.01, 10)

    def couple(self, other: 'GabrielCell'):
        if other not in self.neighbors:
            self.neighbors.append(other)
        if self not in other.neighbors:
            other.neighbors.append(self)

    def neighbor_feedback(self):
        if not self.neighbors:
            return
        avg = np.mean([n.output for n in self.neighbors])
        self.feedback(avg)

"""
Beispiel/Tutorial:
from src.gabriel_cell import GabrielCell

cell = GabrielCell()
out = cell.activate(inp=0.8)
cell.feedback(target=1.5)

cell2 = GabrielCell(psi=0.5)
cell.couple(cell2)
cell2.activate(inp=1.1)
cell.neighbor_feedback()
"""
