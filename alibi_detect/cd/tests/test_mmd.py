import numpy as np
import pytest
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch.mmd import MMDDriftTorch
from alibi_detect.cd.tensorflow.mmd import MMDDriftTF

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
        cd = MMDDrift(x_ref=x_ref, backend=backend)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, MMDDriftTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, MMDDriftTF)
    else:
        assert cd is None
