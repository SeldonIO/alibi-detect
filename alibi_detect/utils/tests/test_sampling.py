from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.sampling import reservoir_sampling

n_X_ref = [5, 10, 100]
n_X = [2, 5, 100]
reservoir_size = [10, 500]
n = [100, 1000]
n_features = 5
tests_sampling = list(product(n_X_ref, n_X, reservoir_size, n))
n_tests = len(tests_sampling)


@pytest.fixture
def update_sampling(request):
    return tests_sampling[request.param]


@pytest.mark.parametrize('update_sampling', list(range(n_tests)), indirect=True)
def test_reservoir_sampling(update_sampling):
    n_X_ref, n_X, reservoir_size, n = update_sampling
    if n_X_ref > reservoir_size:
        return
    X_ref = np.random.rand(n_X_ref * n_features).reshape(n_X_ref, n_features)
    X = np.random.rand(n_X * n_features).reshape(n_X, n_features)
    X_reservoir = reservoir_sampling(X_ref, X, reservoir_size, n)
    n_reservoir = X_reservoir.shape[0]
    assert n_reservoir <= reservoir_size
    if n_reservoir < reservoir_size:
        assert n_reservoir == n_X_ref + n_X
        assert (X_reservoir[:n_X_ref] == X_ref).all()
        assert (X_reservoir[-n_X:] == X).all()
