import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
from alibi_detect.cd.preprocess import UAE, HiddenOutput, pca

n, n_features, n_classes, latent_dim, n_hidden = 100, 10, 5, 2, 7
shape = (n_features,)
X = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')

encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(n_features,)),
        Dense(latent_dim)
    ]
)


tests_uae = [encoder_net, latent_dim]
n_tests_uae = len(tests_uae)


@pytest.fixture
def uae_params(request):
    return tests_uae[request.param]


@pytest.mark.parametrize('uae_params', list(range(n_tests_uae)), indirect=True)
def test_uae(uae_params):
    enc = uae_params
    if isinstance(enc, tf.keras.Sequential):
        encoder_net, enc_dim = enc, None
    elif isinstance(enc, int):
        encoder_net, enc_dim = None, enc
    X_enc = UAE(encoder_net=encoder_net, shape=X.shape[1:], enc_dim=enc_dim)(X)
    assert X_enc.shape == (n, latent_dim)


class Model1(tf.keras.Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.dense1 = Dense(n_hidden)
        self.dense2 = Dense(n_classes, activation='softmax')

    def call(self, x: np.ndarray) -> tf.Tensor:
        x = self.dense1(x)
        return self.dense2(x)


def model2():
    x_in = Input(shape=shape)
    x = Dense(n_hidden)(x_in)
    x_out = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


tests_hidden_output = [
    (1, -2, shape),
    (1, -1, shape),
    (2, -2, None),
    (2, -1, None),
    (2, -1, shape)
]
n_tests_hidden_output = len(tests_hidden_output)


@pytest.fixture
def hidden_output_params(request):
    return tests_hidden_output[request.param]


@pytest.mark.parametrize('hidden_output_params', list(range(n_tests_hidden_output)), indirect=True)
def test_hidden_output(hidden_output_params):
    model, layer, input_shape = hidden_output_params
    model = Model1() if model == 1 else model2()
    X_hidden = HiddenOutput(model=model, layer=layer, input_shape=input_shape)(X)
    if layer == -2:
        assert X_hidden.shape == (n, n_hidden)
    elif layer == -1:
        assert X_hidden.shape == (n, n_classes)


tests_pca = [2, 4]
n_tests_pca = len(tests_pca)


@pytest.fixture
def pca_params(request):
    return tests_pca[request.param]


@pytest.mark.parametrize('pca_params', list(range(n_tests_pca)), indirect=True)
def test_pca(pca_params):
    n_components = pca_params
    X_pca = pca(X, n_components)
    assert X_pca.shape[-1] == n_components
