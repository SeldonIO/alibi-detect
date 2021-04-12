from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from alibi_detect.utils.tensorflow import GaussianRBF, mmd2
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
    x = np.random.random(xshape).astype(np.float32)
    y = np.random.random(yshape).astype(np.float32) * mult

    def metric_fn(x, y):
        return mmd2(x, y, kernel=GaussianRBF(sigma=tf.ones(1))).numpy()

    p_val, dist, dist_permutations = permutation_test(
        x, y, n_permutations=n_permutations, metric=metric_fn
    )
    if mult == 1:
        assert p_val > .2
    elif mult > 1:
        assert p_val <= .2
    assert np.where(dist_permutations >= dist)[0].shape[0] / n_permutations == p_val
