import itertools
import numpy as np
import pytest
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch.mmd import MMDDriftTorch, LinearTimeMMDDriftTorch
from alibi_detect.cd.tensorflow.mmd import MMDDriftTF, LinearTimeMMDDriftTF

n, n_features = 100, 5

tests_backend_mmddrift = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
tests_estimator_mmddrift = ['quad', 'linear', 'Quad', 'Linear', 'cubic']
tests_mmddrift = list(itertools.product(tests_backend_mmddrift, tests_estimator_mmddrift))
n_tests = len(tests_mmddrift)


@pytest.fixture
def mmddrift_params(request):
    return tests_mmddrift[request.param]


@pytest.mark.parametrize('mmddrift_params', list(range(n_tests)), indirect=True)
def test_mmddrift(mmddrift_params):
    backend, estimator = mmddrift_params
    x_ref = np.random.randn(*(n, n_features))

    try:
        cd = MMDDrift(x_ref=x_ref, backend=backend, estimator=estimator)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        if estimator.lower() == 'quad':
            assert isinstance(cd._detector, MMDDriftTorch)
            assert isinstance(cd._detector.n_permutations, int)
        elif estimator.lower() == 'linear':
            assert isinstance(cd._detector, LinearTimeMMDDriftTorch)
            assert hasattr(cd._detector, 'n_permutations')
        else:
            assert cd is None
    elif backend.lower() == 'tensorflow':
        if estimator.lower() == 'quad':
            assert isinstance(cd._detector, MMDDriftTF)
            assert isinstance(cd._detector.n_permutations, int)
        elif estimator.lower() == 'linear':
            assert isinstance(cd._detector, LinearTimeMMDDriftTF)
            assert hasattr(cd._detector, 'n_permutations')
        else:
            assert cd is None
    else:
        assert cd is None
