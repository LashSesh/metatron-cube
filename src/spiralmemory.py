
# src/spiralmemory.py

import numpy as np
from typing import List, Tuple, Optional

class SpiralMemory:
    """
    SpiralMemory: Kodiert Informationen (z.B. Strings, Seeds) als 5D-Punktwolke
    im Spiralraum, führt semantische Bewertung, Mutation und Feedback (Ouroboros)
    durch und gibt bei Konvergenz Exkalibrations-Events aus.
    """

    def __init__(self, alpha: float = 0.1):
        self.memory: List[Tuple[np.ndarray, float]] = []
        self.alpha = alpha  # Anpassungsrate
        self.history = []

    def embed(self, sequence: str) -> np.ndarray:
        base = np.array([ord(c) for c in sequence])
        v = np.zeros(5)
        for i in range(5):
            v[i] = np.sum(base * np.cos(2 * np.pi * (i+1) * np.arange(len(base)) / (len(base) + 5)))
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    def spiralize(self, elements: List[str]) -> List[np.ndarray]:
        return [self.embed(e) for e in elements]

    def psi(self, vi: np.ndarray, vj: np.ndarray) -> float:
        stab = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)
        conv = 1.0 / (1.0 + np.linalg.norm(vi - vj))
        react = np.abs(np.sin(np.sum(vi - vj)))
        return 0.5 * stab + 0.3 * conv + 0.2 * react

    def psi_total(self, points: List[np.ndarray]) -> float:
        return sum(self.psi(points[i], points[i+1]) for i in range(len(points)-1))

    def gradient(self, points: List[np.ndarray]) -> List[np.ndarray]:
        grads = []
        for i in range(len(points)-1):
            diff = points[i+1] - points[i]
            grad = diff / (np.linalg.norm(diff) + 1e-12)
            grads.append(grad)
        grads.append(-grads[-1])
        return grads

    def mutate(self, points: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        return [p + self.alpha * g for p, g in zip(points, grads)]

    def proof_of_resonance(self, psi_old: float, psi_new: float, epsilon: float = 1e-4) -> bool:
        return abs(psi_new - psi_old) < epsilon

    def step(self, elements: List[str], max_iter: int = 30) -> Tuple[List[np.ndarray], float]:
        points = self.spiralize(elements)
        psi_val = self.psi_total(points)
        for _ in range(max_iter):
            grads = self.gradient(points)
            new_points = self.mutate(points, grads)
            new_psi = self.psi_total(new_points)
            self.history.append(new_psi)
            if self.proof_of_resonance(psi_val, new_psi):
                self.memory.append((new_points, new_psi))
                return new_points, new_psi
            points, psi_val = new_points, new_psi
        self.memory.append((points, psi_val))
        return points, psi_val

"""
Beispiel/Tutorial:
from src.spiralmemory import SpiralMemory
sm = SpiralMemory(alpha=0.08)
elements = ['AETHER', 'SILICIUM', 'CYBER', 'QLOGIC']
points, psi = sm.step(elements)
print("5D-Points:", points)
print("ψ:", psi)
"""
