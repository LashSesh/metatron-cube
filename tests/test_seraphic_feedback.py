
# tests/test_seraphic_feedback.py

from src.seraphic_feedback import SeraphicFeedbackModule
import numpy as np

def test_string_input_mapping():
    sfm = SeraphicFeedbackModule()
    v = sfm.map_inputs(['TEST'])
    assert isinstance(v[0], np.ndarray)
    assert v[0].shape == (5,)

def test_numeric_input_mapping():
    sfm = SeraphicFeedbackModule()
    v = sfm.map_inputs([3.0])
    assert v[0].shape == (5,)
    assert np.allclose(np.linalg.norm(v[0]), 1.0, atol=1e-8)

def test_list_input_mapping():
    sfm = SeraphicFeedbackModule()
    v = sfm.map_inputs([[1, 2, 3]])
    assert v[0].shape == (5,)
    assert np.allclose(np.linalg.norm(v[0]), 1.0, atol=1e-8)
