import numpy as np
import pytest
from alibi_detect.cd import LSDDDriftOnline
from alibi_detect.cd.pytorch.lsdd_online import LSDDDriftOnlineTorch
from alibi_detect.cd.tensorflow.lsdd_online import LSDDDriftOnlineTF

n, n_features = 100, 5

tests_lsdddriftonline = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_lsdddriftonline)


@pytest.fixture
def lsdddriftonline_params(request):
    return tests_lsdddriftonline[request.param]


@pytest.mark.parametrize('lsdddriftonline_params', list(range(n_tests)), indirect=True)
def test_lsdddriftonline(lsdddriftonline_params):
    backend = lsdddriftonline_params
    x_ref = np.random.randn(*(n, n_features))

    try:
        cd = LSDDDriftOnline(x_ref=x_ref, ert=25, window_size=5, backend=backend, n_bootstraps=100)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, LSDDDriftOnlineTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, LSDDDriftOnlineTF)
    else:
        assert cd is None
        return None

    # Test predict
    x_t = np.random.randn(n_features)
    t0 = cd.t
    cd.predict(x_t)
    assert cd.t - t0 == 1  # This checks state updated (self.t at least)

    # Test score
    t0 = cd.t
    cd.score(x_t)
    assert cd.t - t0 == 1
