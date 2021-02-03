from itertools import product
import numpy as np
import pytest
from typing import Callable
from alibi_detect.cd import TabularDrift

n = 750
n_categories, n_features = 5, 6

n_cat = [0, 2, n_features]
preprocess = [(None, None)]
correction = ['bonferroni', 'fdr']
update_X_ref = [{'last': 1000}, {'reservoir_sampling': 1000}]
preprocess_X_ref = [True, False]
tests_tabulardrift = list(product(n_cat, preprocess, correction, update_X_ref, preprocess_X_ref))
n_tests = len(tests_tabulardrift)


@pytest.fixture
def tabulardrift_params(request):
    return tests_tabulardrift[request.param]


@pytest.mark.parametrize('tabulardrift_params', list(range(n_tests)), indirect=True)
def test_tabulardrift(tabulardrift_params):
    n_cat, preprocess, correction, update_X_ref, preprocess_X_ref = tabulardrift_params
    n_infer = 2
    preprocess_fn, preprocess_kwargs = preprocess
    if isinstance(preprocess_fn, Callable):
        raise NotImplementedError
    else:
        preprocess_fn, preprocess_kwargs = None, None

    # add categorical variables
    cat_cols = np.random.choice(n_features, size=n_cat, replace=False)
    X_ref = np.random.randn(n * n_features).reshape(n, n_features).astype('float32')
    if n_cat > 0:
        X_ref[:, cat_cols] = np.tile(np.array([np.arange(n / n_categories)] * n_cat).T, (n_categories, 1))

    cd = TabularDrift(
        p_val=.05,
        X_ref=X_ref,
        preprocess_X_ref=preprocess_X_ref,
        update_X_ref=update_X_ref,
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=preprocess_kwargs,
        correction=correction,
        n_infer=n_infer
    )
    X = X_ref.copy()
    preds_batch = cd.predict(X, drift_type='batch', return_p_val=True)
    assert preds_batch['data']['is_drift'] == 0
    k = list(update_X_ref.keys())[0]
    assert cd.n == X.shape[0] + X_ref.shape[0]
    assert cd.X_ref.shape[0] == min(update_X_ref[k], X.shape[0] + X_ref.shape[0])
    assert preds_batch['data']['distance'].min() >= 0.
    if correction == 'bonferroni':
        assert preds_batch['data']['threshold'] == cd.p_val / cd.n_features

    preds_feature = cd.predict(X, drift_type='feature', return_p_val=True)
    assert preds_feature['data']['is_drift'].shape[0] == cd.n_features
    preds_by_feature = (preds_feature['data']['p_val'] < cd.p_val).astype(int)
    assert (preds_feature['data']['is_drift'] == preds_by_feature).all()
    assert preds_feature['data']['threshold'] == cd.p_val
