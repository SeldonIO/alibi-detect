from itertools import product
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
import numpy as np
from alibi_detect.utils.tensorflow import zero_diag, quantile, subset_matrix
from alibi_detect.utils.tensorflow.misc import clone_model


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


def test_subset_matrix():

    mat = tf.range(5)[None, :] * tf.range(5)[:, None]
    inds_0 = [2, 3]
    inds_1 = [2, 1, 4]

    sub_mat = subset_matrix(mat, tf.constant(inds_0), tf.constant(inds_1))
    assert sub_mat.shape == (2, 3)
    for i, ind_0 in enumerate(inds_0):
        for j, ind_1 in enumerate(inds_1):
            assert sub_mat[i, j] == ind_0 * ind_1

    with pytest.raises(ValueError):
        subset_matrix(tf.ones((10, 10, 10)), inds_0, inds_1)
    with pytest.raises(ValueError):
        subset_matrix(tf.ones((10,)), inds_0, inds_1)


n_in, n_out = 10, 5
# sequential model
model_seq = tf.keras.Sequential([InputLayer(n_in, ), Dense(n_out)])

# functional model
inputs = Input(n_in, )
outputs = Dense(n_out)(inputs)
model_func = tf.keras.Model(inputs=inputs, outputs=outputs)


# subclassed model
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(5)

    def call(self, x):
        return self.dense(x)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model_sub = Model()


def test_clone_model():
    model_seq_clone = clone_model(model_seq)
    assert not (model_seq_clone.weights[0] == model_seq.weights[0]).numpy().any()
    model_func_clone = clone_model(model_func)
    assert not (model_func_clone.weights[0] == model_func.weights[0]).numpy().any()
    model_sub_clone = clone_model(model_sub)
    _ = model_sub(tf.zeros((1, 10)))
    _ = model_sub_clone(tf.zeros((1, 10)))
    assert not (model_sub_clone.weights[0] == model_sub.weights[0]).numpy().any()
