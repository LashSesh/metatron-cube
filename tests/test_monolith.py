
# tests/test_monolith.py

from src.monolith import TripolarOperator, OphanKernel, MonolithDecision

def test_tripolar_operator_value():
    op = TripolarOperator(1.0, 2.0, 3.0)
    assert op.value() == 6.0

def test_ophan_kernel_excalibration():
    ops = [TripolarOperator(1, 1, 2) for _ in range(4)]
    kernel = OphanKernel(ops)
    assert kernel.excalibration_ready(threshold=1.0) == True

def test_monolith_decision():
    ops = [TripolarOperator(0.1, 0.1, 0.1) for _ in range(4)]
    dec = MonolithDecision(OphanKernel(ops))
    assert dec.step() == "PENDING"
