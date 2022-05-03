import numpy as np
import pytest
import scipy
from packaging import version
if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift

n, n_test = 500, 200
np.random.seed(0)

n_features = [2]  # TODO - test 1D case once BaseUnivariateDrift updated
tests_cvmdrift = list(n_features)
n_tests = len(tests_cvmdrift)


@pytest.fixture
def cvmdrift_params(request):
    return tests_cvmdrift[request.param]


@pytest.mark.skipif(version.parse(scipy.__version__) < version.parse('1.7.0'),
                    reason="Requires scipy version >= 1.7.0")
@pytest.mark.parametrize('cvmdrift_params', list(range(n_tests)), indirect=True)
def test_cvmdrift(cvmdrift_params):
    n_feat = cvmdrift_params

    # Reference data
    x_ref = np.random.normal(0, 1, size=(n, n_feat)).squeeze()  # squeeze to test vec input in 1D case

    # Instantiate detector
    cd = CVMDrift(x_ref=x_ref, p_val=0.05)

    # Test predict on reference data
    x_h0 = x_ref.copy()
    preds = cd.predict(x_h0, return_p_val=True)
    assert preds['data']['is_drift'] == 0 and (preds['data']['p_val'] >= cd.p_val).any()

    # Test predict on heavily drifted data
    x_h1 = np.random.normal(2, 2, size=(n, n_feat)).squeeze()
    preds = cd.predict(x_h1, drift_type='batch')
    assert preds['data']['is_drift'] == 1
    assert preds['data']['distance'].min() >= 0.
