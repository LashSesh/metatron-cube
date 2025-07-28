
# tests/test_field_vector.py

from src.field_vector import FieldVector
import numpy as np

def test_norm_and_normalize():
    v = FieldVector([1, 2, 3, 4, 5])
    nrm = v.norm()
    assert nrm > 0
    normed = v.normalize()
    assert np.allclose(np.linalg.norm(normed), 1.0)

def test_similarity():
    v1 = FieldVector([1, 0, 0, 0, 0])
    v2 = FieldVector([1, 1, 0, 0, 0])
    sim = v1.similarity(v2.vec)
    assert 0.6 < sim < 1.1

def test_trm2_update():
    v = FieldVector([1, 0, 0, 0, 0], omega=0.1)
    out1 = v.trm2_update([1, -1, 0.5, 0, 0.3])
    out2 = v.trm2_update([1, 0, 0, 0, 0.3])
    assert isinstance(out1, float)
    assert isinstance(out2, float)
    assert v.phi != 0.0

def test_add_and_scale():
    v = FieldVector([1, 2, 3, 4, 5])
    v2 = v.add([0, 1, 0, 0, 0])
    v3 = v.scale(2.0)
    assert v2.vec[1] == 3
    assert np.allclose(v3.vec, np.array([2, 4, 6, 8, 10]))
