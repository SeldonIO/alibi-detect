import numpy as np
import pytest
from alibi_detect.cd import FETDrift
from itertools import product

n, n_test = 500, 200
np.random.seed(0)

alternative = ['less', 'greater', 'two-sided']
n_features = [2]  # TODO - test 1D case once BaseUnivariateDrift updated
tests_fetdrift = list(product(alternative, n_features))
n_tests = len(tests_fetdrift)


@pytest.fixture
def fetdrift_params(request):
    return tests_fetdrift[request.param]


@pytest.mark.parametrize('fetdrift_params', list(range(n_tests)), indirect=True)
def test_fetdrift(fetdrift_params):
    alternative, n_feat = fetdrift_params

    # Reference data
    p_h0 = 0.5
    x_ref = np.random.choice([0, 1], (n, n_feat), p=[1 - p_h0, p_h0]).squeeze()  # squeeze to test vec input in 1D case

    # Instantiate detector
    cd = FETDrift(x_ref=x_ref, p_val=0.05, alternative=alternative)

    # Test predict on reference data
    x_h0 = x_ref.copy()
    preds = cd.predict(x_h0, return_p_val=True)
    assert preds['data']['is_drift'] == 0 and (preds['data']['p_val'] >= cd.p_val).any()

    # Test predict on heavily drifted data
    if alternative == 'less' or alternative == 'two-sided':
        p_h1 = 0.2
        x_h1 = np.random.choice([0, 1], (n_test, n_feat), p=[1 - p_h1, p_h1]).squeeze()
        preds = cd.predict(x_h1)
        assert preds['data']['is_drift'] == 1
    if alternative == 'greater' or alternative == 'two-sided':
        p_h1 = 0.8
        x_h1 = np.random.choice([0, 1], (n_test, n_feat), p=[1 - p_h1, p_h1]).squeeze()
        preds = cd.predict(x_h1)
        assert preds['data']['is_drift'] == 1

    # Check odds ratio
    ref_1s = np.sum(x_ref, axis=0)
    ref_0s = len(x_ref) - ref_1s
    test1_1s = np.sum(x_h1, axis=0)
    test1_0s = len(x_h1) - test1_1s
    odds_ratio = (test1_1s / test1_0s) / (ref_1s / ref_0s)
    for f in range(n_feat):
        np.testing.assert_allclose(preds['data']['distance'][f],
                                   odds_ratio[f], rtol=1e-05)


@pytest.mark.parametrize('bad_value', [0.5, 1.5, 2, -1, np.nan])
def test_fetdrift_rejects_non_binary(bad_value):
    # Non-binary batches must raise even when the values would truncate to 0/1
    # under an int cast (see https://github.com/SeldonIO/alibi-detect/issues/962).
    x_ref = np.random.choice([0, 1], (n, 2))
    cd = FETDrift(x_ref=x_ref, p_val=0.05)
    x_bad = np.full((n_test, 2), bad_value)
    with pytest.raises(ValueError, match="must consist of only"):
        cd.predict(x_bad)


@pytest.mark.parametrize('good_dtype', [np.int64, bool, np.float64])
def test_fetdrift_accepts_binary_dtypes(good_dtype):
    # Binary data in any reasonable dtype must keep working.
    x_ref = np.random.choice([0, 1], (n, 2))
    cd = FETDrift(x_ref=x_ref, p_val=0.05)
    x_good = np.random.choice([0, 1], (n_test, 2)).astype(good_dtype)
    preds = cd.predict(x_good)
    assert preds['data']['is_drift'] in (0, 1)
