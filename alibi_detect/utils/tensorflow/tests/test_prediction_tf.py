import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from typing import Tuple, Union
from alibi_detect.utils.tensorflow import predict_batch

n, n_features, n_classes, latent_dim = 100, 10, 5, 2
x = np.zeros((n, n_features), dtype=np.float32)


class MyModel(tf.keras.Model):
    def __init__(self, multi_out: bool = False):
        super(MyModel, self).__init__()
        self.dense = Dense(n_classes, activation='softmax')
        self.multi_out = multi_out

    def call(self, x: np.ndarray) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        out = self.dense(x)
        if not self.multi_out:
            return out
        else:
            return out, out


AutoEncoder = tf.keras.Sequential(
    [
        InputLayer(input_shape=(n_features,)),
        Dense(latent_dim),
        Dense(n_features)
    ]
)


def id_fn(x: Union[np.ndarray, tf.Tensor, list]) -> Union[np.ndarray, tf.Tensor]:
    if isinstance(x, list):
        return np.concatenate(x, axis=0)
    else:
        return x


# model, batch size, dtype, preprocessing function, list as input
tests_predict = [
    (MyModel(multi_out=False), 2, np.float32, None, False),
    (MyModel(multi_out=False), int(1e10), np.float32, None, False),
    (MyModel(multi_out=False), int(1e10), tf.float32, None, False),
    (MyModel(multi_out=True), int(1e10), tf.float32, None, False),
    (MyModel(multi_out=False), int(1e10), np.float32, id_fn, False),
    (AutoEncoder, 2, np.float32, None, False),
    (AutoEncoder, int(1e10), np.float32, None, False),
    (AutoEncoder, int(1e10), tf.float32, None, False),
    (id_fn, 2, np.float32, None, False),
    (id_fn, 2, tf.float32, None, False),
    (id_fn, 2, np.float32, id_fn, True),
]
n_tests = len(tests_predict)


@pytest.fixture
def predict_batch_params(request):
    return tests_predict[request.param]


@pytest.mark.parametrize('predict_batch_params', list(range(n_tests)), indirect=True)
def test_predict_batch(predict_batch_params):
    model, batch_size, dtype, preprocess_fn, to_list = predict_batch_params
    x_batch = [x] if to_list else x
    preds = predict_batch(x_batch, model, batch_size=batch_size, preprocess_fn=preprocess_fn, dtype=dtype)
    if isinstance(preds, tuple):
        preds = preds[0]
    assert preds.dtype == dtype
    if isinstance(model, tf.keras.Sequential) or hasattr(model, '__name__') and model.__name__ == 'id_fn':
        assert preds.shape == x.shape
    elif isinstance(model, tf.keras.Model):
        assert preds.shape == (n, n_classes)
