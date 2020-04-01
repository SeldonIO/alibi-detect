from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
from typing import Callable
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.preprocess import hidden_output, uae

n, n_hidden, n_classes = 500, 10, 5


def mymodel(shape):
    x_in = Input(shape=shape)
    x = Dense(n_hidden)(x_in)
    x_out = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


n_features = [10]
n_enc = [None, 3]
preprocess = [
    (None, None),
    (hidden_output, {'model': None, 'layer': -1}),
    (uae, {'encoder_net': None})
]
chunk_size = [None, 500]
update_X_ref = [{'last': 750}, {'reservoir_sampling': 750}]
n_permutations = [5]
tests_mmddrift = list(product(n_features, n_enc, preprocess, chunk_size, n_permutations, update_X_ref))
n_tests = len(tests_mmddrift)


@pytest.fixture
def mmd_params(request):
    return tests_mmddrift[request.param]


@pytest.mark.parametrize('mmd_params', list(range(n_tests)), indirect=True)
def test_mmd(mmd_params):
    n_features, n_enc, preprocess, chunk_size, n_permutations, update_X_ref = mmd_params
    np.random.seed(0)
    X_ref = np.random.randn(n * n_features).reshape(n, n_features).astype('float32')
    preprocess_fn, preprocess_kwargs = preprocess
    if isinstance(preprocess_fn, Callable):
        if preprocess_fn.__name__ == 'uae' and n_features > 1 and isinstance(n_enc, int):
            encoder_net = tf.keras.Sequential(
                [
                    InputLayer(input_shape=(n_features,)),
                    Dense(n_enc)
                ]
            )
            preprocess_kwargs['encoder_net'] = encoder_net
        elif preprocess_fn.__name__ == 'hidden_output':
            model = mymodel((n_features,))
            preprocess_kwargs['model'] = model
        else:
            preprocess_fn, preprocess_kwargs = None, None
    else:
        preprocess_fn, preprocess_kwargs = None, None

    cd = MMDDrift(
        p_val=.05,
        X_ref=X_ref,
        update_X_ref=update_X_ref,
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=preprocess_kwargs,
        chunk_size=chunk_size,
        n_permutations=n_permutations
    )
    X = X_ref.copy()
    preds = cd.predict(X, return_p_val=True)
    assert preds['data']['is_drift'] == 0 and preds['data']['p_val'] >= cd.p_val
    k = list(update_X_ref.keys())[0]
    assert cd.n == X.shape[0] + X_ref.shape[0]
    assert cd.X_ref.shape[0] == min(update_X_ref[k], X.shape[0] + X_ref.shape[0])

    X_h1 = np.random.randn(n * n_features).reshape(n, n_features).astype('float32')
    mu, sigma = 5, 5
    X_h1 = sigma * X_h1 + mu
    preds = cd.predict(X_h1, return_p_val=True)
    assert preds['data']['is_drift'] == 1 and preds['data']['p_val'] < cd.p_val
