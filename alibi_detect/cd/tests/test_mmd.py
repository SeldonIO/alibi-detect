import numpy as np
import pytest
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch.mmd import MMDDriftTorch
from alibi_detect.cd.tensorflow.mmd import MMDDriftTF
from alibi_detect.utils.frameworks import has_keops
if has_keops:
    from alibi_detect.cd.keops.mmd import MMDDriftKeops

n, n_features = 100, 5

tests_mmddrift = ['tensorflow', 'pytorch', 'keops', 'PyToRcH', 'mxnet']
n_tests = len(tests_mmddrift)


@pytest.fixture
def mmddrift_params(request):
    return tests_mmddrift[request.param]


@pytest.mark.parametrize('mmddrift_params', list(range(n_tests)), indirect=True)
def test_mmddrift(mmddrift_params):
    backend = mmddrift_params
    x_ref = np.random.randn(*(n, n_features)).astype('float32')

    try:
        cd = MMDDrift(x_ref=x_ref, backend=backend)
    except (NotImplementedError, ImportError):
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, MMDDriftTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, MMDDriftTF)
    elif backend.lower() == 'keops' and has_keops:
        assert isinstance(cd._detector, MMDDriftKeops)
    else:
        assert cd is None
