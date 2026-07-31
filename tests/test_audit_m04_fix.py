import torch
import numpy as np
from glassbox.sr.operations.meta_ops import MetaArithmeticExtended

def test_m04_soft_div_parity():
    """M-04: MetaArithmeticExtended soft division term uses sqrt form matching eval.h / export_pytorch."""
    m = MetaArithmeticExtended()
    
    # Force weights to favor division term (beta=2.0, gamma=-1.0)
    m.beta.data.fill_(2.0)
    m.gamma.data.fill_(-1.0)
    
    x = torch.tensor([2.0, -3.0, 0.5, -1.0], dtype=torch.float64)
    y = torch.tensor([3.0, -2.0, 0.0, 4.0], dtype=torch.float64)
    
    # Compute expected soft division term: x / sqrt(1 + y^2)
    expected_div = x / torch.sqrt(1.0 + torch.square(y))
    
    # Compute weights from MetaArithmeticExtended
    d_add = (m.beta - 1.0)**2 + (m.gamma - 1.0)**2
    d_mul = (m.beta - 2.0)**2 + (m.gamma - 1.0)**2
    d_div = (m.beta - 2.0)**2 + (m.gamma + 1.0)**2
    d_sub = (m.beta - 1.0)**2 + (m.gamma + 1.0)**2
    
    logits = torch.stack([-d_add, -d_mul, -d_div, -d_sub])
    weights = torch.nn.functional.softmax(logits * 5.0, dim=0)
    
    res_add = x + y
    res_sub = x - y
    res_mul = x * y
    res_div = expected_div
    
    expected_result = (
        weights[0] * res_add +
        weights[1] * res_mul +
        weights[2] * res_div +
        weights[3] * res_sub
    )
    expected_result = torch.clamp(expected_result, -100, 100)
    
    actual_result = m(x, y)
    
    assert torch.allclose(actual_result, expected_result), f"Mismatch: actual {actual_result} vs expected {expected_result}"
