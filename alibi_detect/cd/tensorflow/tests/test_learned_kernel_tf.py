from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense
from typing import Union
from alibi_detect.cd.tensorflow.learned_kernel import LearnedKernelDriftTF

n = 100


class MyKernel(tf.keras.Model):  # TODO: Support then test models using keras functional API
    def __init__(self, n_features: int):
        super().__init__()
        self.config = {'n_features': n_features}
        self.dense = Dense(20)

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.einsum('ji,ki->jk', self.dense(x), self.dense(y))

    def get_config(self) -> dict:
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# test List[Any] inputs to the detector
def identity_fn(x: Union[np.array, list]) -> np.array:
    if isinstance(x, list):
        return np.array(x)
    else:
        return x


p_val = [.05]
n_features = [4]
train_size = [.5]
preprocess_batch = [None, identity_fn]
update_x_ref = [None, {'last': 1000}, {'reservoir_sampling': 1000}]
tests_lkdrift = list(product(p_val, n_features, train_size, preprocess_batch, update_x_ref))
n_tests = len(tests_lkdrift)


@pytest.fixture
def lkdrift_params(request):
    return tests_lkdrift[request.param]


@pytest.mark.parametrize('lkdrift_params', list(range(n_tests)), indirect=True)
def test_lkdrift(lkdrift_params):
    p_val, n_features, train_size, preprocess_batch, update_x_ref = lkdrift_params

    np.random.seed(0)
    tf.random.set_seed(0)

    kernel = MyKernel(n_features)
    x_ref = np.random.randn(*(n, n_features))
    x_test1 = np.ones_like(x_ref)
    to_list = False
    if preprocess_batch is not None:
        to_list = True
        x_ref = [_ for _ in x_ref]
        update_x_ref = None

    cd = LearnedKernelDriftTF(
        x_ref=x_ref,
        kernel=kernel,
        p_val=p_val,
        update_x_ref=update_x_ref,
        train_size=train_size,
        preprocess_batch_fn=preprocess_batch,
        batch_size=3,
        epochs=1
    )

    x_test0 = x_ref.copy()
    preds_0 = cd.predict(x_test0)
    assert cd.n == len(x_test0) + len(x_ref)
    assert preds_0['data']['is_drift'] == 0

    if to_list:
        x_test1 = [_ for _ in x_test1]
    preds_1 = cd.predict(x_test1)
    assert cd.n == len(x_test1) + len(x_test0) + len(x_ref)
    assert preds_1['data']['is_drift'] == 1

    assert preds_0['data']['distance'] < preds_1['data']['distance']
