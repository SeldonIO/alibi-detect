from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from alibi_detect.od import LLR

shape = (100, 6)
X_train = np.zeros(shape, dtype=np.int32)
X_train[:, ::2] = 1
X_test = np.zeros(shape, dtype=np.int32)
X_test[:, ::2] = 2
X_val = np.concatenate([X_train[:50], X_test[:50]])

input_dim = 3
hidden_dim = 10


def loss_fn(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    y = tf.one_hot(tf.cast(y, tf.int32), input_dim)
    return tf.nn.softmax_cross_entropy_with_logits(y, x, axis=-1)


def likelihood_fn(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    return -loss_fn(y, x)


threshold = [None, 5.]
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
    logits = Dense(input_dim, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    od = LLR(threshold=threshold, sequential=True, model=model, log_prob=likelihood_fn)

    assert od.threshold == threshold
    assert od.meta == {'name': 'LLR', 'detector_type': 'offline', 'data_type': None}

    od.fit(
        X_train,
        loss_fn=loss_fn,
        mutate_fn_kwargs={'rate': .2, 'feature_range': (0, 2)},
        epochs=5,
        verbose=False
    )
