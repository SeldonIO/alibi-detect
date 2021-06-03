import torch
from alibi_detect.utils.pytorch import zero_diag


def test_zero_diag():
    ones = torch.ones(10, 10)
    ones_zd = zero_diag(ones)
    assert ones_zd.shape == (10, 10)
    assert float(ones_zd.trace()) == 0
    assert float(ones_zd.sum()) == 90
