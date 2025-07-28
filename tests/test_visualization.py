
# tests/test_visualization.py

from src.visualization import plot_history, plot_multiple, plot_fieldvectors

def test_plot_history_runs():
    plot_history([0, 1, 0.5, 0.8, 1.0], title="Test", ylabel="Value")

def test_plot_multiple_runs():
    plot_multiple([[0,1,2,3],[1,2,3,4]], labels=["A", "B"], title="Vergleich")

def test_plot_fieldvectors_runs():
    plot_fieldvectors([[1,0,0,0,0],[0,1,0,0,0]], title="Vectors")
