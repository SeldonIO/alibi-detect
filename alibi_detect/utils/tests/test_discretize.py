from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.discretizer import Discretizer

x = np.random.rand(10, 4)
n_features = x.shape[1]
feature_names = [str(_) for _ in range(n_features)]

categorical_features = [[], [1, 3]]
percentiles = [list(np.arange(25, 100, 25)), list(np.arange(10, 100, 10))]
tests = list(product(categorical_features, percentiles))
n_tests = len(tests)


@pytest.fixture
def cats_and_percentiles(request):
    cat, perc = tests[request.param]
    return cat, perc


@pytest.mark.parametrize('cats_and_percentiles', list(range(n_tests)), indirect=True)
def test_discretizer(cats_and_percentiles):
    cat, perc = cats_and_percentiles
    disc = Discretizer(x, cat, feature_names, perc)
    to_disc = list(disc.names.keys())
    assert len(to_disc) == (x.shape[1] - len(cat))
    x_disc = disc.discretize(x)
    for k, v in disc.names.items():
        assert len(v) <= len(perc) + 1
        assert callable(disc.lambdas[k])
        assert (x_disc[:, k].min() == 0).all()
        assert (x_disc[:, k].max() == len(perc)).all()

    for i in range(x.shape[1]):
        if i not in to_disc:
            assert (x_disc[:, i] == x[:, i]).all()
