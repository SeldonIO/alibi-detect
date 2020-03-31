from itertools import product
import numpy as np
import pytest
from alibi_detect.cd.utils import update_reference

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
