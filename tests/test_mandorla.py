
# tests/test_mandorla.py

from src.mandorla import MandorlaField
import numpy as np

def test_add_input_and_resonance():
    mf = MandorlaField(threshold=0.99)
    v1 = np.ones(5)
    v2 = np.ones(5) * 0.99
    mf.add_input(v1)
    mf.add_input(v2)
    res = mf.calc_resonance()
    assert res > 0.95

def test_decision_trigger():
    mf = MandorlaField(threshold=0.98)
    v1 = np.ones(5)
    v2 = np.ones(5)
    mf.add_input(v1)
    mf.add_input(v2)
    assert mf.decision_trigger() is True

def test_clear_inputs():
    mf = MandorlaField()
    mf.add_input(np.ones(5))
    mf.clear_inputs()
    assert len(mf.inputs) == 0
