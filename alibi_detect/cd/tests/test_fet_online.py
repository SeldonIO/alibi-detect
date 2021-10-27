import numpy as np
import pytest
from itertools import product
from alibi_detect.cd import FETDriftOnline

n, n_features = 100, 1
n_bootstraps = 1000

window_size = [10, [10, 20]]
alternative = ['two-sided']  # , 'less', 'greater']  # TODO - not implemented yet

tests_fetdriftonline = list(product(window_size, alternative))
n_tests = len(tests_fetdriftonline)


@pytest.fixture
def fetdriftonline_params(request):
    return tests_fetdriftonline[request.param]


@pytest.mark.parametrize('fetdriftonline_params', list(range(n_tests)), indirect=True)
def test_fetdriftonline(fetdriftonline_params):
    window_size, alternative = fetdriftonline_params

    # Reference data
    np.random.seed(0)
    p_h0 = 0.5
    p_h1 = 0.3
    x_ref = np.random.choice(2, n, p=[1 - p_h0, p_h0])
    stream_h1 = (np.random.choice(2, p=[1 - p_h1, p_h1]) for _ in range(int(1e4)))

    # Instantiate detector
    cd = FETDriftOnline(x_ref=x_ref, ert=25, window_size=window_size, n_bootstraps=n_bootstraps)  # TODO - alternative

    # Test predict
    x_t = np.array([next(stream_h1)])
    t0 = cd.t
    cd.predict(x_t)
    assert cd.t - t0 == 1  # This checks state updated (self.t at least)

    # Test score
    x_t = np.array([next(stream_h1)])
    t0 = cd.t
    cd.score(x_t)
    assert cd.t - t0 == 1
