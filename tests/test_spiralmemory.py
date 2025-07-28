
# tests/test_spiralmemory.py

from src.spiralmemory import SpiralMemory

def test_embedding_and_spiralize():
    sm = SpiralMemory()
    points = sm.spiralize(['TEST', 'CODE'])
    assert len(points) == 2
    assert points[0].shape == (5,)

def test_psi_and_gradient():
    sm = SpiralMemory()
    p1, p2 = sm.embed('X'), sm.embed('Y')
    psi_val = sm.psi(p1, p2)
    assert -1 <= psi_val <= 2
    grads = sm.gradient([p1, p2])
    assert len(grads) == 2

def test_step_exkalibration():
    sm = SpiralMemory(alpha=0.09)
    elements = ['ALPHA', 'BETA', 'GAMMA']
    points, psi = sm.step(elements, max_iter=20)
    assert isinstance(points, list)
    assert len(points) == 3
    assert isinstance(psi, float)
