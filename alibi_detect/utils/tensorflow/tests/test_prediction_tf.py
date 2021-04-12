import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.models.tensorflow import AE
from alibi_detect.utils.tensorflow import predict_batch

n, n_features, n_classes, latent_dim = 100, 10, 5, 2
X = np.zeros((n, n_features))


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = Dense(n_classes, activation='softmax')

    def call(self, x: np.ndarray) -> tf.Tensor:
        return self.dense(x)


model = MyModel()

encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(n_features,)),
        Dense(latent_dim)
    ]
)
decoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim,)),
        Dense(n_features)
    ]
)
AutoEncoder = AE(encoder_net, decoder_net)

# model, batch size, dtype
tests_predict = [
    (model, 2, np.float32),
    (model, int(1e10), np.float32),
    (model, int(1e10), tf.float32),
    (AutoEncoder, 2, np.float32),
    (AutoEncoder, int(1e10), np.float32)
]
n_tests = len(tests_predict)


@pytest.fixture
def predict_batch_params(request):
    return tests_predict[request.param]


@pytest.mark.parametrize('predict_batch_params', list(range(n_tests)), indirect=True)
def test_predict_batch(predict_batch_params):
    model, batch_size, dtype = predict_batch_params
    preds = predict_batch(X, model, batch_size=batch_size, dtype=dtype)
    assert preds.dtype == dtype
    if isinstance(model, AE):
        assert preds.shape == X.shape
    elif isinstance(model, tf.keras.Model):
        assert preds.shape == (n, n_classes)
