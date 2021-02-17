from itertools import product
import numpy as np
import pytest
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from alibi_detect.cd import ClassifierDrift
from alibi_detect.utils.metrics import accuracy

n = 100


def mymodel(shape):
    x_in = Input(shape=shape)
    x = Dense(20, activation=tf.nn.relu)(x_in)
    x_out = Dense(2, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


def f1_adj(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, np.round(y_pred))


threshold = [.6]
n_features = [4]
metric_fns = [accuracy, f1_adj]
n_folds = [None, 2]
train_size = [.5]
update_X_ref = [None, {'last': 1000}, {'reservoir_sampling': 1000}]
tests_clfdrift = list(product(threshold, n_features, metric_fns, n_folds,
                              train_size, update_X_ref))
n_tests = len(tests_clfdrift)


@pytest.fixture
def clfdrift_params(request):
    return tests_clfdrift[request.param]


@pytest.mark.parametrize('clfdrift_params', list(range(n_tests)), indirect=True)
def test_clfdrift(clfdrift_params):
    threshold, n_features, metric_fn, n_folds, train_size, update_X_ref = clfdrift_params

    np.random.seed(0)
    tf.random.set_seed(0)

    model = mymodel((n_features,))
    X_ref = np.random.randn(*(n, n_features))
    X_test0 = X_ref.copy()
    X_test1 = np.ones_like(X_ref)

    cd = ClassifierDrift(
        threshold=threshold,
        model=model,
        X_ref=X_ref,
        update_X_ref=update_X_ref,
        train_size=train_size,
        n_folds=n_folds,
        metric_fn=metric_fn,
        batch_size=1
    )

    preds_0 = cd.predict(X_test0)
    assert preds_0['data'][f'{metric_fn.__name__}'] <= threshold
    assert cd.n == X_test0.shape[0] + X_ref.shape[0]
    assert preds_0['data']['is_drift'] == 0

    preds_1 = cd.predict(X_test1)
    assert preds_1['data'][f'{metric_fn.__name__}'] > threshold
    assert cd.n == X_test1.shape[0] + X_test0.shape[0] + X_ref.shape[0]
    assert preds_1['data']['is_drift'] == 1

    assert cd.meta['params']['metric_fn'] == metric_fn.__name__
