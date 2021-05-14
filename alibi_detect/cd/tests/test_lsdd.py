import numpy as np
import pytest
from alibi_detect.cd import LSDDDrift
from alibi_detect.cd.pytorch.lsdd import LSDDDriftTorch
from alibi_detect.cd.tensorflow.lsdd import LSDDDriftTF

n, n_features = 100, 5

tests_lsdddrift = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_lsdddrift)


@pytest.fixture
def lsdddrift_params(request):
    return tests_lsdddrift[request.param]


@pytest.mark.parametrize('lsdddrift_params', list(range(n_tests)), indirect=True)
def test_lsdddrift(lsdddrift_params):
    backend = lsdddrift_params
    x_ref = np.random.randn(*(n, n_features))

    try:
        cd = LSDDDrift(x_ref=x_ref, backend=backend)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, LSDDDriftTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, LSDDDriftTF)
    else:
        assert cd is None
