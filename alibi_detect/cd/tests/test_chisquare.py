from itertools import product
import numpy as np
import pytest
from typing import Callable
from alibi_detect.cd import ChiSquareDrift

n_categories, n_features, n_tiles = 5, 6, 20
X_ref = np.tile(np.array([np.arange(n_categories)] * n_features).T, (n_tiles, 1))
np.random.shuffle(X_ref)

preprocess = [(None, None)]
correction = ['bonferroni', 'fdr']
update_X_ref = [{'last': 1000}, {'reservoir_sampling': 1000}]
preprocess_X_ref = [True, False]
tests_chisquaredrift = list(product(preprocess, correction, update_X_ref, preprocess_X_ref))
n_tests = len(tests_chisquaredrift)


@pytest.fixture
def chisquaredrift_params(request):
    return tests_chisquaredrift[request.param]


@pytest.mark.parametrize('chisquaredrift_params', list(range(n_tests)), indirect=True)
def test_chisquaredrift(chisquaredrift_params):
    preprocess, correction, update_X_ref, preprocess_X_ref = chisquaredrift_params
    n_infer = 2
    preprocess_fn, preprocess_kwargs = preprocess
    if isinstance(preprocess_fn, Callable):
        raise NotImplementedError
    else:
        preprocess_fn, preprocess_kwargs = None, None

    cd = ChiSquareDrift(
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
