from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM
from alibi_detect.od import LLR
from alibi_detect.version import __version__

input_dim = 5
hidden_dim = 20

shape = (1000, 6)
X_train = np.zeros(shape, dtype=np.int32)
X_train[:, ::2] = 1
X_test = np.zeros(shape, dtype=np.int32)
X_test[:, ::2] = 2
X_val = np.concatenate([X_train[:50], X_test[:50]])


def loss_fn(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    y = tf.one_hot(tf.cast(y, tf.int32), input_dim)
    return tf.nn.softmax_cross_entropy_with_logits(y, x, axis=-1)


def likelihood_fn(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    return - loss_fn(y, x)


threshold = [None]
threshold_perc = [50.]
return_instance_score = [True, False]
return_feature_score = [True, False]
outlier_type = ['instance', 'feature']
tests = list(product(threshold, threshold_perc, return_instance_score,
                     return_feature_score, outlier_type))
n_tests = len(tests)


@pytest.fixture
def llr_params(request):
    return tests[request.param]


@pytest.mark.parametrize('llr_params', list(range(n_tests)), indirect=True)
def test_llr(llr_params):
    # LLR parameters
    threshold, threshold_perc, return_instance_score, return_feature_score, outlier_type = llr_params

    # define model and detector
    inputs = Input(shape=(shape[-1] - 1,), dtype=tf.int32)
    x = tf.one_hot(tf.cast(inputs, tf.int32), input_dim)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    logits = Dense(input_dim, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    od = LLR(threshold=threshold, sequential=True, model=model, log_prob=likelihood_fn)

    assert od.threshold == threshold
    assert od.meta == {'name': 'LLR', 'detector_type': 'offline', 'data_type': None, 'version': __version__}

    od.fit(
        X_train,
        loss_fn=loss_fn,
        mutate_fn_kwargs={'rate': .5, 'feature_range': (0, input_dim)},
        epochs=1,
        verbose=False
    )

    od.infer_threshold(X_val, threshold_perc=threshold_perc)
    # iscore_test = od.score(X_test)[1]
    # iscore_train = od.score(X_train)[1]
    # assert (iscore_test > iscore_train).all()

    od_preds = od.predict(X_test,
                          return_instance_score=return_instance_score,
                          return_feature_score=return_feature_score,
                          outlier_type=outlier_type)

    assert od_preds['meta'] == od.meta
    if outlier_type == 'instance':
        assert od_preds['data']['is_outlier'].shape == (X_test.shape[0],)
        if return_instance_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score']
                                                            > od.threshold).astype(int).sum()
    elif outlier_type == 'feature':
        assert od_preds['data']['is_outlier'].shape == (X_test.shape[0], X_test.shape[1] - 1)
        if return_feature_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['feature_score']
                                                            > od.threshold).astype(int).sum()

    if return_feature_score:
        assert od_preds['data']['feature_score'].shape == (X_test.shape[0], X_test.shape[1] - 1)
    else:
        assert od_preds['data']['feature_score'] is None

    if return_instance_score:
        assert od_preds['data']['instance_score'].shape == (X_test.shape[0],)
    else:
        assert od_preds['data']['instance_score'] is None
