
# src/field_vector.py

import numpy as np

class FieldVector:
    """
    Universelle 5D (oder n-dim) Vektor- und Resonanz-Utility:
    - Vektor-Arithmetik
    - Normierung
    - Kosinus-Ähnlichkeit
    - Projektion/Addition/Skalierung
    - Multipolar TRM2-Resonanz/Decision-Update (emergent, dynamisch, Blueprint-perfect!)
    """

    def __init__(self, data, omega=0.0):
        self.vec = np.array(data, dtype=np.float64)
        self.n = len(self.vec)
        self.omega = omega  # Grundfrequenz (z.B. für Oszillator)
        self.phi = 0.0      # Aktuelle Phase (für TRM)
        self.history = []

    def norm(self):
        return np.linalg.norm(self.vec)

    def normalize(self):
        nrm = self.norm()
        if nrm == 0: return self.vec
        self.vec = self.vec / nrm
        return self.vec

    def similarity(self, other):
        o = np.array(other)
        return float(np.dot(self.vec, o) / ((np.linalg.norm(self.vec) * np.linalg.norm(o)) + 1e-12))

    def add(self, other):
        return FieldVector(self.vec + np.array(other))

    def scale(self, s):
        return FieldVector(self.vec * s)

    def trm2_update(self, inputs, kappas=None, thetas=None, dt=1.0):
        n = self.n
        inputs = np.array(inputs)
        kappas = np.ones(n) if kappas is None else np.array(kappas)
        thetas = np.linspace(0, 2*np.pi, n, endpoint=False) if thetas is None else np.array(thetas)
        dphi = self.omega
        for i in range(n):
            dphi += kappas[i] * inputs[i] * np.sin(thetas[i] - self.phi)
        self.phi += dphi * dt
        self.history.append(self.phi)
        return np.sin(self.phi)

    def as_array(self):
        return np.array(self.vec)

    def __repr__(self):
        return f"FieldVector({self.vec}, omega={self.omega}, phi={self.phi})"
