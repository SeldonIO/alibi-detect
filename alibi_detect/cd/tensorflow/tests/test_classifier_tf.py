from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from typing import Union
from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF

n = 100


def mymodel(shape, softmax: bool = True):
    x_in = Input(shape=shape)
    x = Dense(20, activation=tf.nn.relu)(x_in)
    x = Dense(2)(x)
    if softmax:
        x = tf.nn.softmax(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x)


# test List[Any] inputs to the detector
def identity_fn(x: Union[np.array, list]) -> np.array:
    if isinstance(x, list):
        return np.array(x)
    else:
        return x


p_val = [.05]
n_features = [4]
preds_type = ['probs', 'logits']
binarize_preds = [True, False]
n_folds = [None, 2]
train_size = [.5]
preprocess_batch = [None, identity_fn]
update_x_ref = [None, {'last': 1000}, {'reservoir_sampling': 1000}]
tests_clfdrift = list(product(p_val, n_features, preds_type, binarize_preds, n_folds,
                              train_size, preprocess_batch, update_x_ref))
n_tests = len(tests_clfdrift)


@pytest.fixture
def clfdrift_params(request):
    return tests_clfdrift[request.param]


@pytest.mark.parametrize('clfdrift_params', list(range(n_tests)), indirect=True)
def test_clfdrift(clfdrift_params):
    p_val, n_features, preds_type, binarize_preds, n_folds, \
        train_size, preprocess_batch, update_x_ref = clfdrift_params

    np.random.seed(0)
    tf.random.set_seed(0)

    model = mymodel((n_features,), softmax=(preds_type == 'probs'))
    x_ref = np.random.randn(*(n, n_features))
    x_test1 = np.ones_like(x_ref)
    to_list = False
    if preprocess_batch is not None:
        to_list = True
        x_ref = [_ for _ in x_ref]
        update_x_ref = None

    cd = ClassifierDriftTF(
        x_ref=x_ref,
        model=model,
        p_val=p_val,
        update_x_ref=update_x_ref,
        train_size=train_size,
        n_folds=n_folds,
        preds_type=preds_type,
        binarize_preds=binarize_preds,
        preprocess_batch_fn=preprocess_batch,
        batch_size=1
    )

    x_test0 = x_ref.copy()
    preds_0 = cd.predict(x_test0)
    assert cd.n == len(x_test0) + len(x_ref)
    assert preds_0['data']['is_drift'] == 0
    assert preds_0['data']['distance'] >= 0

    if to_list:
        x_test1 = [_ for _ in x_test1]
    preds_1 = cd.predict(x_test1)
    assert cd.n == len(x_test1) + len(x_test0) + len(x_ref)
    assert preds_1['data']['is_drift'] == 1
    assert preds_1['data']['distance'] >= 0

    assert preds_0['data']['distance'] < preds_1['data']['distance']
    assert cd.meta['params']['preds_type'] == preds_type
    assert cd.meta['params']['binarize_preds '] == binarize_preds
