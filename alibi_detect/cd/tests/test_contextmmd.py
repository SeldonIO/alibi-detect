import numpy as np
import pytest
from alibi_detect.cd import ContextMMDDrift
from alibi_detect.cd.pytorch.context_aware import ContextMMDDriftTorch
from alibi_detect.cd.tensorflow.context_aware import ContextMMDDriftTF

n, n_features = 100, 5

tests_context_mmddrift = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_context_mmddrift)


@pytest.fixture
def context_mmddrift_params(request):
    return tests_context_mmddrift[request.param]


@pytest.mark.parametrize('context_mmddrift_params', list(range(n_tests)), indirect=True)
def test_context_mmddrift(context_mmddrift_params):
    backend = context_mmddrift_params
    c_ref = np.random.randn(*(n, 1))
    x_ref = c_ref + np.random.randn(*(n, n_features))

    try:
        cd = ContextMMDDrift(x_ref=x_ref, c_ref=c_ref, backend=backend)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, ContextMMDDriftTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, ContextMMDDriftTF)
    else:
        assert cd is None
