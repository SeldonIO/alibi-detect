from itertools import product
import numpy as np
import pytest
from alibi_detect.cd import TabularDrift

n = 750
n_categories, n_features = 5, 6

n_cat = [0, 2, n_features]
categories_per_feature = [None, int, list]
correction = ['bonferroni', 'fdr']
update_x_ref = [{'last': 1000}, {'reservoir_sampling': 1000}]
preprocess_x_ref = [True, False]
new_categories = [True, False]
tests_tabulardrift = list(product(n_cat, categories_per_feature, correction,
                                  update_x_ref, preprocess_x_ref, new_categories))
n_tests = len(tests_tabulardrift)


@pytest.fixture
def tabulardrift_params(request):
    return tests_tabulardrift[request.param]


@pytest.mark.parametrize('tabulardrift_params', list(range(n_tests)), indirect=True)
def test_tabulardrift(tabulardrift_params):
    n_cat, categories_per_feature, correction, \
        update_x_ref, preprocess_x_ref, new_categories = tabulardrift_params
    np.random.seed(0)
    # add categorical variables
    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    if n_cat > 0:
        cat_cols = np.random.choice(n_features, size=n_cat, replace=False)
        x_ref[:, cat_cols] = np.tile(np.array([np.arange(n_categories)] * n_cat).T, (n // n_categories, 1))
        if categories_per_feature is None:
            categories_per_feature = {f: None for f in cat_cols}
        elif categories_per_feature == int:
            categories_per_feature = {f: n_categories for f in cat_cols}
        elif categories_per_feature == list:
            categories_per_feature = {f: list(np.arange(n_categories)) for f in cat_cols}
    else:
        categories_per_feature = None

    cd = TabularDrift(
        x_ref=x_ref,
        p_val=.05,
        categories_per_feature=categories_per_feature,
        preprocess_x_ref=preprocess_x_ref,
        update_x_ref=update_x_ref,
        correction=correction,
    )
    x = x_ref.copy()
    if new_categories and n_cat > 0:
        x[:, cat_cols] = x[:, cat_cols] + 1
    preds_batch = cd.predict(x, drift_type='batch', return_p_val=True)
    if new_categories and n_cat > 0:
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
