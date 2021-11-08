import numpy as np
import pytest
from alibi_detect.cd import MMDDriftOnline
from alibi_detect.cd.pytorch.mmd_online import MMDDriftOnlineTorch
from alibi_detect.cd.tensorflow.mmd_online import MMDDriftOnlineTF

n, n_features = 100, 5

tests_mmddriftonline = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_mmddriftonline)


@pytest.fixture
def mmddriftonline_params(request):
    return tests_mmddriftonline[request.param]


@pytest.mark.parametrize('mmddriftonline_params', list(range(n_tests)), indirect=True)
def test_mmddriftonline(mmddriftonline_params):
    backend = mmddriftonline_params
    x_ref = np.random.randn(*(n, n_features))

    # Instantiate and check detector class
    try:
        cd = MMDDriftOnline(x_ref=x_ref, ert=25, window_size=5, backend=backend, n_bootstraps=100)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, MMDDriftOnlineTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, MMDDriftOnlineTF)
    else:
        assert cd is None
        return

    # Test predict
    x_t = np.random.randn(n_features)
    t0 = cd.t
    cd.predict(x_t)
    assert cd.t - t0 == 1  # This checks state updated (self.t at least)

    # Test score
    t0 = cd.t
    cd.score(x_t)
    assert cd.t - t0 == 1
