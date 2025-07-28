
# src/visualization.py

import matplotlib.pyplot as plt

def plot_history(history, title="History", ylabel="Value"):
    plt.figure()
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.show()

def plot_decision_points(history, threshold=0.98, title="Decision Points"):
    plt.figure()
    plt.plot(history, label='Resonance')
    decisions = [i for i, v in enumerate(history) if v > threshold]
    plt.scatter(decisions, [history[i] for i in decisions], marker='o', label='Decision', zorder=5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Resonance")
    plt.legend()
    plt.show()

def plot_vector(vec, title="FieldVector"):
    import numpy as np
    plt.figure()
    data = vec if isinstance(vec, (list, tuple)) else getattr(vec, "vec", None)
    if data is not None:
        data = list(data)
        plt.bar(range(len(data)), data)
        plt.title(title)
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.show()
