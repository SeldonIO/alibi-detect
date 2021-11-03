import numpy as np
import pytest
from itertools import product
from alibi_detect.cd import CVMDriftOnline

n, n_features = 100, 1
n_bootstraps = 1000

window_sizes = [[10], [10, 20]]
batch_size = [None, int(n_bootstraps/4)]

tests_cvmdriftonline = list(product(window_sizes, batch_size))
n_tests = len(tests_cvmdriftonline)


@pytest.fixture
def cvmdriftonline_params(request):
    return tests_cvmdriftonline[request.param]


@pytest.mark.parametrize('cvmdriftonline_params', list(range(n_tests)), indirect=True)
def test_cvmdriftonline(cvmdriftonline_params):
    window_sizes, batch_size = cvmdriftonline_params

    # Reference data
    np.random.seed(0)
    x_ref = np.random.randn(*(n, n_features))

    # Instantiate detector
    cd = CVMDriftOnline(x_ref=x_ref, ert=25, window_sizes=window_sizes,
                        n_bootstraps=n_bootstraps, batch_size=batch_size)

    # Test predict
    x_t = np.random.randn(1, n_features)
    t0 = cd.t
    cd.predict(x_t)
    assert cd.t - t0 == 1  # This checks state updated (self.t at least)

    # Test score
    t0 = cd.t
    cd.score(x_t)
    assert cd.t - t0 == 1
