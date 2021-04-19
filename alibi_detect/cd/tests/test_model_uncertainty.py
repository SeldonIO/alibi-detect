import numpy as np
import pytest
from functools import partial
from itertools import product
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Softmax
import torch
import torch.nn as nn
from alibi_detect.cd import ClassifierUncertaintyDrift

n = 300
# TODO: Test multiclass


def tf_model(n_features, prediction_type):
    x_in = Input(shape=(n_features,))
    x = Dense(20, activation=tf.nn.relu)(x_in)
    x = Dense(2)(x)
    if prediction_type == 'probs':
        x = Softmax()(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x)


class PtModel(nn.Module):
    def __init__(self, n_features, prediction_type):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)
        self.prediction_type = prediction_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        x = self.dense2(x)
        if self.prediction_type == 'probs':
            x = nn.Softmax()(x)
        return x


def dumb_model(x, prediction_type):
    if prediction_type == 'probs':
        probs = 1 / (1 + np.exp(-x.mean(axis=-1)))
        return np.stack([probs, 1-probs], axis=1)
    elif prediction_type == 'logits':
        logits = np.mean(x, axis=-1)
        return np.stack([logits, -logits], axis=1)
    else:
        raise ValueError()


def other_model(prediction_type):
    if prediction_type == 'probs':
        return partial(dumb_model, prediction_type='probs')
    elif prediction_type == 'logits':
        return partial(dumb_model, prediction_type='logits')


def gen_model(n_features, backend, prediction_type):
    if backend == 'tensorflow':
        return tf_model(n_features, prediction_type)
    elif backend == 'pytorch':
        return PtModel(n_features, prediction_type)
    elif backend is None:
        return other_model(prediction_type)


p_val = [.05]
backend = ['tensorflow', 'pytorch', None]
n_features = [16]
prediction_type = ['probs', 'logits']
uncertainty_type = ['entropy', 'margin']
update_x_ref = [None, {'last': 1000}, {'reservoir_sampling': 1000}]
tests_clfuncdrift = list(product(p_val, backend, n_features, prediction_type, uncertainty_type, update_x_ref))
n_tests = len(tests_clfuncdrift)


@pytest.fixture
def clfuncdrift_params(request):
    return tests_clfuncdrift[request.param]


@pytest.mark.parametrize('clfuncdrift_params', list(range(n_tests)), indirect=True)
def test_clfuncdrift(clfuncdrift_params):
    p_val, backend, n_features, prediction_type, uncertainty_type, update_x_ref = clfuncdrift_params

    np.random.seed(0)
    tf.random.set_seed(0)

    model = gen_model(n_features, backend, prediction_type)
    x_ref = np.random.randn(*(n, n_features)).astype(np.float32)
    x_test0 = x_ref.copy()
    x_test1 = np.ones_like(x_ref)

    cd = ClassifierUncertaintyDrift(
        x_ref=x_ref,
        model=model,
        p_val=p_val,
        backend=backend,
        update_x_ref=update_x_ref,
        prediction_type=prediction_type,
        uncertainty_type=uncertainty_type,
        margin_width=0.2,
        batch_size=10
    )

    preds_0 = cd.predict(x_test0)
    assert cd._detector.n == x_test0.shape[0] + x_ref.shape[0]
    assert preds_0['data']['is_drift'] == 0
    assert preds_0['data']['distance'] >= 0

    preds_1 = cd.predict(x_test1)
    assert cd._detector.n == x_test1.shape[0] + x_test0.shape[0] + x_ref.shape[0]
    if not preds_1['data']['is_drift'] == 1:
        breakpoint()
    assert preds_1['data']['is_drift'] == 1
    assert preds_1['data']['distance'] >= 0

    assert preds_0['data']['distance'] < preds_1['data']['distance']
    # assert cd.meta['params']['soft_preds'] == soft_preds
