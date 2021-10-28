import numpy as np
import pytest
from alibi_detect.cd import FETDrift

n, n_features = 100, 1

alternative = ['less', 'greater']

tests_fetdrift = list(alternative)
n_tests = len(tests_fetdrift)


@pytest.fixture
def fetdrift_params(request):
    return tests_fetdrift[request.param]


@pytest.mark.parametrize('fetdrift_params', list(range(n_tests)), indirect=True)
def test_fetdrift(fetdrift_params):
    alternative = fetdrift_params

    # Reference data
    np.random.seed(0)
    p_h0 = 0.5
    p_h1 = 0.3
    x_ref = np.random.choice(2, n, p=[1 - p_h0, p_h0])

    # Instantiate detector
    cd = FETDrift(x_ref=x_ref, p_val=0.05, alternative=alternative)

    # Test predict
    x = np.random.choice(2, n, p=[1 - p_h1, p_h1])
    cd.predict(x)
