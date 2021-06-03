from itertools import product
import pytest
import torch
import numpy as np
from alibi_detect.utils.pytorch import zero_diag, quantile


def test_zero_diag():
    ones = torch.ones(10, 10)
    ones_zd = zero_diag(ones)
    assert ones_zd.shape == (10, 10)
    assert float(ones_zd.trace()) == 0
    assert float(ones_zd.sum()) == 90


type = [6, 7, 8]
sorted = [True, False]
tests_quantile = list(product(type, sorted))
n_tests_quantile = len(tests_quantile)


@pytest.fixture
def quantile_params(request):
    return tests_quantile[request.param]


@pytest.mark.parametrize('quantile_params', list(range(n_tests_quantile)), indirect=True)
def test_quantile(quantile_params):
    type, sorted = quantile_params

    sample = (0.5+torch.arange(1e6))/1e6
    if not sorted:
        sample = sample[torch.randperm(len(sample))]

    np.testing.assert_almost_equal(quantile(sample, 0.001, type=type, sorted=sorted), 0.001, decimal=6)
    np.testing.assert_almost_equal(quantile(sample, 0.999, type=type, sorted=sorted), 0.999, decimal=6)

    assert quantile(torch.ones(100), 0.42, type=type, sorted=sorted) == 1
    with pytest.raises(ValueError):
        quantile(torch.ones(10), 0.999, type=type, sorted=sorted)
    with pytest.raises(ValueError):
        quantile(torch.ones(100, 100), 0.5, type=type, sorted=sorted)
