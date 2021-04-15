from itertools import product
import numpy as np
import pytest
from alibi_detect.cd import ChiSquareDrift

n_categories, n_features, n_tiles = 5, 6, 20
x_ref = np.tile(np.array([np.arange(n_categories)] * n_features).T, (n_tiles, 1))
np.random.shuffle(x_ref)

categories_per_feature = [
    None,
    {f: n_categories for f in range(n_features)},
    {f: list(np.arange(n_categories)) for f in range(n_features)}
]
preprocess = [None]
correction = ['bonferroni', 'fdr']
update_x_ref = [{'last': 1000}, {'reservoir_sampling': 1000}]
preprocess_x_ref = [True, False]
new_categories = [True, False]
tests_chisquaredrift = list(product(categories_per_feature, preprocess, correction,
                                    update_x_ref, preprocess_x_ref, new_categories))
n_tests = len(tests_chisquaredrift)


@pytest.fixture
def chisquaredrift_params(request):
    return tests_chisquaredrift[request.param]


@pytest.mark.parametrize('chisquaredrift_params', list(range(n_tests)), indirect=True)
def test_chisquaredrift(chisquaredrift_params):
    categories_per_feature, preprocess_fn, correction, \
        update_x_ref, preprocess_x_ref, new_categories = chisquaredrift_params

    cd = ChiSquareDrift(
        x_ref=x_ref,
        p_val=.05,
        categories_per_feature=categories_per_feature,
        preprocess_x_ref=preprocess_x_ref,
        update_x_ref=update_x_ref,
        preprocess_fn=preprocess_fn,
        correction=correction,
    )
    x = x_ref.copy()
    if new_categories:
        x = x + 1
    preds_batch = cd.predict(x, drift_type='batch', return_p_val=True)
    if new_categories:
        assert preds_batch['data']['is_drift'] == 1
    else:
        assert preds_batch['data']['is_drift'] == 0
    k = list(update_x_ref.keys())[0]
    assert cd.n == x.shape[0] + x_ref.shape[0]
    assert cd.x_ref.shape[0] == min(update_x_ref[k], x.shape[0] + x_ref.shape[0])
    assert preds_batch['data']['distance'].min() >= 0.
    if correction == 'bonferroni':
        assert preds_batch['data']['threshold'] == cd.p_val / cd.n_features

    preds_feature = cd.predict(x, drift_type='feature', return_p_val=True)
    assert preds_feature['data']['is_drift'].shape[0] == cd.n_features
    preds_by_feature = (preds_feature['data']['p_val'] < cd.p_val).astype(int)
    assert (preds_feature['data']['is_drift'] == preds_by_feature).all()
    assert preds_feature['data']['threshold'] == cd.p_val
