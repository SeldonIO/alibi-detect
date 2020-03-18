from itertools import product
import numpy as np
import pytest
from alibi_detect.cd.utils import update_reference, fdr

n = [3, 50]
n_features = [1, 10]
update_method = [
    None,
    'last',
    'reservoir_sampling'
]
tests_update = list(product(n, n_features, update_method))
n_tests_update = len(tests_update)


@pytest.fixture
def update_params(request):
    return tests_update[request.param]


@pytest.mark.parametrize('update_params', list(range(n_tests_update)), indirect=True)
def test_update_reference(update_params):
    n, n_features, update_method = update_params
    n_ref = np.random.randint(1, n)
    n_test = np.random.randint(1, 2 * n)
    X_ref = np.random.rand(n_ref * n_features).reshape(n_ref, n_features)
    X = np.random.rand(n_test * n_features).reshape(n_test, n_features)
    if update_method in ['last', 'reservoir_sampling']:
        update_method = {update_method: n}
    X_ref_new = update_reference(X_ref, X, n, update_method)

    assert X_ref_new.shape[0] <= n
    if isinstance(update_method, dict):
        if list(update_method.keys())[0] == 'last':
            assert (X_ref_new[-1] == X[-1]).all()


q_val = [.05, .1, .25]
n_p = 1000
p_vals = [
    {'is_below': True, 'p_val': np.zeros(n_p)},
    {'is_below': False, 'p_val': np.zeros(n_p)}
]
tests_fdr = list(product(q_val, p_vals))
n_tests_fdr = len(tests_fdr)


@pytest.fixture
def fdr_params(request):
    return tests_fdr[request.param]


@pytest.mark.parametrize('fdr_params', list(range(n_tests_fdr)), indirect=True)
def test_fdr(fdr_params):
    q_val, p_vals = fdr_params
    if p_vals['is_below'] and p_vals['p_val'].max() == 0:
        p_val = p_vals['p_val'] + q_val - 1e-5
    elif not p_vals['is_below'] and p_vals['p_val'].max() == 0:
        p_val = p_vals['p_val'] + q_val
    else:
        p_val = p_vals['p_val'].copy()
    below_threshold = fdr(p_val, q_val)
    assert below_threshold == p_vals['is_below']
