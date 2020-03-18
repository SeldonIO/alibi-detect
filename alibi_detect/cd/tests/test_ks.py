from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
from typing import Callable
from alibi_detect.cd import KSDrift
from alibi_detect.cd.preprocess import hidden_output, pca, uae

n, n_hidden, n_classes = 750, 10, 5


def mymodel(shape):
    x_in = Input(shape=shape)
    x = Dense(n_hidden)(x_in)
    x_out = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


n_features = [1, 10]
n_enc = [None, 3]
preprocess = [
    (None, None),
    (hidden_output, {'model': None, 'layer': -1}),
    (pca, {'n_components': None}),
    (uae, {'encoder_net': None})
]
alternative = ['two-sided', 'less', 'greater']
correction = ['bonferroni', 'fdr']
update_X_ref = [{'last': 1000}, {'reservoir_sampling': 1000}]
tests_ksdrift = list(product(n_features, n_enc, preprocess, alternative, correction, update_X_ref))
n_tests = len(tests_ksdrift)


@pytest.fixture
def ksdrift_params(request):
    return tests_ksdrift[request.param]


@pytest.mark.parametrize('ksdrift_params', list(range(n_tests)), indirect=True)
def test_ksdrift(ksdrift_params):
    n_features, n_enc, preprocess, alternative, correction, update_X_ref = ksdrift_params
    np.random.seed(0)
    X_ref = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')
    n_infer = 2
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
        elif preprocess_fn.__name__ == 'pca' and isinstance(n_enc, int):
            if n_enc < n_features:
                preprocess_kwargs['n_components'] = n_enc
                n_infer = n_enc
            else:
                preprocess_fn, preprocess_kwargs = None, None
        else:
            preprocess_fn, preprocess_kwargs = None, None
    else:
        preprocess_fn, preprocess_kwargs = None, None

    cd = KSDrift(
        p_val=.05,
        X_ref=X_ref,
        update_X_ref=update_X_ref,
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=preprocess_kwargs,
        correction=correction,
        alternative=alternative,
        n_infer=n_infer
    )
    X = X_ref.copy()
    preds_batch = cd.predict(X, drift_type='batch', return_feature_score=True)
    assert preds_batch['data']['is_drift'] == 0
    k = list(update_X_ref.keys())[0]
    assert cd.n == X.shape[0] + X_ref.shape[0]
    assert cd.X_ref.shape[0] == min(update_X_ref[k], X.shape[0] + X_ref.shape[0])

    preds_feature = cd.predict(X, drift_type='feature', return_feature_score=True)
    assert preds_feature['data']['is_drift'].shape[0] == cd.n_features
    preds_by_feature = (preds_feature['data']['feature_score'] < cd.p_val).astype(int)
    assert (preds_feature['data']['is_drift'] == preds_by_feature).all()

    mult = 100
    X_high, X_low = X_ref.copy() * mult, X_ref.copy() / mult
    preds_batch = cd.predict(X_high, drift_type='batch')
    if alternative != 'less':
        assert preds_batch['data']['is_drift'] == 1
    preds_batch = cd.predict(X_low, drift_type='batch')
    if alternative != 'greater':
        assert preds_batch['data']['is_drift'] == 1
