from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from alibi_detect.models.trainer import trainer

N, F = 100, 2
x = np.random.rand(N, F).astype(np.float32)
y = np.concatenate([np.zeros((N, 1)), np.ones((N, 1))], axis=1).astype(np.float32)

inputs = tf.keras.Input(shape=(x.shape[1],))
outputs = tf.keras.layers.Dense(F, activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
check_model_weights = model.weights[0].numpy()

X_train = [x]
y_train = [None, y]
loss_fn_kwargs = [None, {'from_logits': False}]
verbose = [False, True]

tests = list(product(X_train, y_train, loss_fn_kwargs, verbose))
n_tests = len(tests)


@pytest.fixture
def trainer_params(request):
    X_train, y_train, loss_fn_kwargs, verbose = tests[request.param]
    return X_train, y_train, loss_fn_kwargs, verbose


@pytest.mark.parametrize('trainer_params', list(range(n_tests)), indirect=True)
def test_trainer(trainer_params):
    X_train, y_train, loss_fn_kwargs, verbose = trainer_params
    trainer(model,
            categorical_crossentropy,
            X_train,
            y_train=y_train,
            loss_fn_kwargs=loss_fn_kwargs,
            epochs=2,
            verbose=verbose
            )
    assert (model.weights[0].numpy() != check_model_weights).any()
