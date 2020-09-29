import dask.array as da
from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.distance import maximum_mean_discrepancy
from alibi_detect.utils.statstest import fdr, permutation_test

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
    below_threshold, thresholds = fdr(p_val, q_val)
    assert below_threshold == p_vals['is_below']
    assert isinstance(thresholds, (np.ndarray, float))


n_features = [2]
n_instances = [(100, 100), (100, 75)]
n_permutations = [10]
mult = [1, 5]
tests_permutation = list(product(n_features, n_instances, n_permutations, mult))
n_tests_permutation = len(tests_permutation)


@pytest.fixture
def permutation_params(request):
    return tests_permutation[request.param]


@pytest.mark.parametrize('permutation_params', list(range(n_tests_permutation)), indirect=True)
def test_permutation(permutation_params):
    n_features, n_instances, n_permutations, mult = permutation_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    np.random.seed(0)
    x = np.random.random(xshape).astype('float32')
    y = np.random.random(yshape).astype('float32') * mult
    xda = da.from_array(x, chunks=xshape)
    yda = da.from_array(y, chunks=yshape)

    kwargs = {'sigma': np.array([1.])}
    p_val = permutation_test(x, y, n_permutations=n_permutations,
                             metric=maximum_mean_discrepancy, **kwargs)
    p_val_da = permutation_test(xda, yda, n_permutations=n_permutations,
                                metric=maximum_mean_discrepancy, **kwargs)

    if mult == 1:
        assert p_val > .2 and p_val_da > .2
    elif mult > 1:
        assert p_val <= .2 and p_val_da <= .2
