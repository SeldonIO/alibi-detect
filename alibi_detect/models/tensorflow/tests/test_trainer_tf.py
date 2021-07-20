from functools import partial
from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from alibi_detect.models.tensorflow.trainer import trainer
from alibi_detect.utils.tensorflow.data import TFDataset

N, F = 100, 2
x = np.random.rand(N, F).astype(np.float32)
y = np.concatenate([np.zeros((N, 1)), np.ones((N, 1))], axis=1).astype(np.float32)

inputs = tf.keras.Input(shape=(x.shape[1],))
outputs = tf.keras.layers.Dense(F, activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
check_model_weights = model.weights[0].numpy()


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    return x


X_train = [x]
y_train = [None, y]
dataset = [partial(TFDataset, batch_size=10), None]
loss_fn_kwargs = [None, {'from_logits': False}]
preprocess = [preprocess_fn, None]
verbose = [False, True]

tests = list(product(X_train, y_train, dataset, loss_fn_kwargs, preprocess, verbose))
n_tests = len(tests)


@pytest.fixture
def trainer_params(request):
    x_train, y_train, dataset, loss_fn_kwargs, preprocess, verbose = tests[request.param]
    return x_train, y_train, dataset, loss_fn_kwargs, preprocess, verbose


@pytest.mark.parametrize('trainer_params', list(range(n_tests)), indirect=True)
def test_trainer(trainer_params):
    x_train, y_train, dataset, loss_fn_kwargs, preprocess, verbose = trainer_params
    if dataset is not None and y_train is not None:
        ds = dataset(x_train, y_train)
    else:
        ds = None
    trainer(model, categorical_crossentropy, x_train, y_train=y_train, dataset=ds,
            loss_fn_kwargs=loss_fn_kwargs, preprocess_fn=preprocess, epochs=2, verbose=verbose)
    assert (model.weights[0].numpy() != check_model_weights).any()
