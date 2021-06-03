from itertools import product
import pytest
import tensorflow as tf
import numpy as np
from alibi_detect.utils.tensorflow import zero_diag, quantile


def test_zero_diag():
    ones = tf.ones((10, 10))
    ones_zd = zero_diag(ones)
    assert ones_zd.shape == (10, 10)
    assert float(tf.linalg.trace(ones_zd)) == 0
    assert float(tf.reduce_sum(ones_zd)) == 90


type = [6, 7, 8]
sorted = [True, False]
tests_quantile = list(product(type, sorted))
n_tests_quantile = len(tests_quantile)


@pytest.fixture
def quantile_params(request):
    return tests_quantile[request.param]


@pytest.mark.parametrize('quantile_params', list(range(n_tests_quantile)), indirect=True)
def test_quantile(quantile_params):
    type, sorted = quantile_params

    sample = (0.5+tf.range(1e6))/1e6
    if not sorted:
        sample = tf.random.shuffle(sample)

    np.testing.assert_almost_equal(quantile(sample, 0.001, type=type, sorted=sorted), 0.001, decimal=6)
    np.testing.assert_almost_equal(quantile(sample, 0.999, type=type, sorted=sorted), 0.999, decimal=6)

    assert quantile(tf.ones((100,)), 0.42, type=type, sorted=sorted) == 1
    with pytest.raises(ValueError):
        quantile(tf.ones((10,)), 0.999, type=type, sorted=sorted)
    with pytest.raises(ValueError):
        quantile(tf.ones((100, 100)), 0.5, type=type, sorted=sorted)
