import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.utils.prediction import predict_batch
from alibi_detect.models.autoencoder import AE

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

# model, proba, return_class, shape
tests_predict = [
    (model, True, False, None),
    (model, False, True, None),
    (model, False, False, (n, n_classes)),
    (AutoEncoder, False, False, None),
    (AutoEncoder, True, False, None)
]
n_tests = len(tests_predict)


@pytest.fixture
def update_predict_batch(request):
    return tests_predict[request.param]


@pytest.mark.parametrize('update_predict_batch', list(range(n_tests)), indirect=True)
def test_predict_batch(update_predict_batch):
    model, proba, return_class, shape = update_predict_batch
    preds = predict_batch(model, X, proba=proba, return_class=return_class, shape=shape)
    if isinstance(model, AE):
        assert preds.shape == X.shape
    elif isinstance(model, tf.keras.Model) and proba:
        assert preds.shape == (n, n_classes)
    elif isinstance(model, tf.keras.Model) and not proba and return_class:
        assert preds.shape == (n,)
    elif isinstance(model, tf.keras.Model) and shape:
        assert preds.shape == shape
