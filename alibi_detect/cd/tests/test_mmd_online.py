import numpy as np
import pytest
from alibi_detect.cd import MMDDriftOnline
from alibi_detect.cd.pytorch.mmd_online import MMDDriftOnlineTorch
from alibi_detect.cd.tensorflow.mmd_online import MMDDriftOnlineTF

n, n_features = 100, 5

tests_mmddrift = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_mmddrift)


@pytest.fixture
def mmddrift_params(request):
    return tests_mmddrift[request.param]


@pytest.mark.parametrize('mmddrift_params', list(range(n_tests)), indirect=True)
def test_mmddrift(mmddrift_params):
    backend = mmddrift_params
    x_ref = np.random.randn(*(n, n_features))

    try:
        cd = MMDDriftOnline(x_ref=x_ref, ert=20, window_size=5, backend=backend, n_bootstraps=100)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, MMDDriftOnlineTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, MMDDriftOnlineTF)
    else:
        assert cd is None
