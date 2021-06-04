import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
from alibi_detect.cd.tensorflow import UAE, HiddenOutput

n, n_features, n_classes, latent_dim = 100, 10, 5, 2
X_uae = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')

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
    X_enc = UAE(encoder_net=encoder_net, shape=X_uae.shape[1:], enc_dim=enc_dim)(X_uae)
    assert X_enc.shape == (n, latent_dim)


dim1, dim2, n_hidden = 2, 3, 7
n_features = dim1 * dim2
shape = (dim1, dim2)
X_h = np.random.rand(n * n_features).reshape((n,) + shape).astype('float32')


class Model1(tf.keras.Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.dense1 = Dense(n_hidden)
        self.dense2 = Dense(n_classes, activation='softmax')

    def call(self, x: np.ndarray) -> tf.Tensor:
        x = self.dense1(x)
        return self.dense2(x)


def model2():
    x_in = Input(shape=(dim1, dim2))
    x = Dense(n_hidden)(x_in)
    x_out = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


tests_hidden_output = [
    (1, -2, shape, True), (1, -2, shape, False),
    (1, -1, shape, True), (1, -1, shape, False),
    (2, -2, None, True), (2, -2, None, False),
    (2, -1, None, True), (2, -1, None, False),
    (2, -1, shape, True), (2, -1, shape, False)
]
n_tests_hidden_output = len(tests_hidden_output)


@pytest.fixture
def hidden_output_params(request):
    return tests_hidden_output[request.param]


@pytest.mark.parametrize('hidden_output_params', list(range(n_tests_hidden_output)), indirect=True)
def test_hidden_output(hidden_output_params):
    model, layer, input_shape, flatten = hidden_output_params
    print(model, layer, input_shape, flatten)
    model = Model1() if model == 1 else model2()
    X_hidden = HiddenOutput(model=model, layer=layer, input_shape=input_shape, flatten=flatten)(X_h)
    if layer == -2:
        assert_shape = (n, dim1, n_hidden)
    elif layer == -1:
        assert_shape = (n, dim1, n_classes)
    if flatten:
        assert_shape = (assert_shape[0],) + (np.prod(assert_shape[1:]),)
    assert X_hidden.shape == assert_shape
