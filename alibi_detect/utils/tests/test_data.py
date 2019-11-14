from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.data import create_outlier_batch, Bunch

N, F = 1000, 4
X = np.random.rand(N, F)
y = np.zeros(N,)
y[:int(.5 * N)] = 1

n_samples = [50, 100]
perc_outlier = [10, 50]

tests = list(product(n_samples, perc_outlier))
n_tests = len(tests)


@pytest.fixture
def batch_params(request):
    return tests[request.param]


@pytest.mark.parametrize('batch_params', list(range(n_tests)), indirect=True)
def test_outlier_batch(batch_params):
    n_samples, perc_outlier = batch_params
    batch = create_outlier_batch(X, y, n_samples, perc_outlier)
    assert isinstance(batch, Bunch)
    assert batch.data.shape == (n_samples, F)
    assert batch.target.shape == (n_samples,)
    assert batch.target_names == ['normal', 'outlier']
    assert int(100 * batch.target.sum() / n_samples) == perc_outlier
