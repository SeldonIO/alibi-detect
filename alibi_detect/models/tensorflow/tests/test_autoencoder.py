import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.models.tensorflow.autoencoder import AE, AEGMM, VAE, VAEGMM, Seq2Seq, EncoderLSTM, DecoderLSTM
from alibi_detect.models.tensorflow.losses import loss_aegmm, loss_vaegmm
from alibi_detect.models.tensorflow.trainer import trainer

input_dim = 784
latent_dim = 50

encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(input_dim,)),
        Dense(128, activation=tf.nn.relu),
        Dense(latent_dim, activation=None)
    ]
)

decoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim,)),
        Dense(128, activation=tf.nn.relu),
        Dense(input_dim, activation=tf.nn.sigmoid)
    ]
)

ae = AE(encoder_net, decoder_net)
vae = VAE(encoder_net, decoder_net, latent_dim)
tests = [ae, vae]


@pytest.fixture
def tf_v_ae_mnist(request):
    # load and preprocess MNIST data
    (X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
    X = X_train.reshape(60000, input_dim)[:1000]  # only train on 1000 instances
    X = X.astype(np.float32)
    X /= 255

    # init model, predict with untrained model, train and predict with trained model
    model = request.param
    X_recon_untrained = model(X).numpy()
    model_weights = model.weights[1].numpy().copy()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=5)
    X_recon = model(X).numpy()
    assert (model_weights != model.weights[1].numpy()).any()
    assert np.sum((X - X_recon_untrained)**2) > np.sum((X - X_recon)**2)


@pytest.mark.parametrize('tf_v_ae_mnist', tests, indirect=True)
def test_ae_vae(tf_v_ae_mnist):
    pass


n_gmm = 1
gmm_density_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim + 2,)),
        Dense(10, activation=tf.nn.relu),
        Dense(n_gmm, activation=tf.nn.softmax)
    ]
)

aegmm = AEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm)
vaegmm = VAEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm, latent_dim)
tests = [(aegmm, loss_aegmm), (vaegmm, loss_vaegmm)]
n_tests = len(tests)


@pytest.fixture
def tf_v_aegmm_mnist(request):
    # load and preprocess MNIST data
    (X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
    X = X_train.reshape(60000, input_dim)[:1000]  # only train on 1000 instances
    X = X.astype(np.float32)
    X /= 255

    # init model, predict with untrained model, train and predict with trained model
    model, loss_fn = tests[request.param]
    X_recon_untrained, z, gamma = model(X)
    assert X_recon_untrained.shape == X.shape
    assert z.shape[1] == latent_dim + 2
    assert gamma.shape[1] == n_gmm
    model_weights = model.weights[1].numpy().copy()
    trainer(model, loss_fn, X, epochs=5, verbose=False, batch_size=1000)
    assert (model_weights != model.weights[1].numpy()).any()


@pytest.mark.parametrize('tf_v_aegmm_mnist', list(range(n_tests)), indirect=True)
def test_aegmm_vaegmm(tf_v_aegmm_mnist):
    pass


seq_len = 10
tests_seq2seq = [(DecoderLSTM(latent_dim, 1, None), 1),
                 (DecoderLSTM(latent_dim, 2, None), 2)]
n_tests = len(tests_seq2seq)


@pytest.fixture
def tf_seq2seq_sine(request):
    # create artificial sine time series
    X = np.sin(np.linspace(-50, 50, 10000)).astype(np.float32)

    # init model
    decoder_net_, n_features = tests_seq2seq[request.param]
    encoder_net = EncoderLSTM(latent_dim)
    threshold_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(seq_len, latent_dim)),
            Dense(10, activation=tf.nn.relu)
        ]
    )
    model = Seq2Seq(encoder_net, decoder_net_, threshold_net, n_features)

    # reshape data
    shape = (-1, seq_len, n_features)
    y = np.roll(X, -1, axis=0).reshape(shape)
    X = X.reshape(shape)

    # predict with untrained model, train and predict with trained model
    X_recon_untrained = model(X)
    assert X_recon_untrained.shape == X.shape
    model_weights = model.weights[1].numpy().copy()
    trainer(model, tf.keras.losses.mse, X, y_train=y, epochs=2, verbose=False, batch_size=64)
    X_recon = model(X).numpy()
    assert (model_weights != model.weights[1].numpy()).any()
    assert np.sum((X - X_recon_untrained)**2) > np.sum((X - X_recon)**2)


@pytest.mark.parametrize('tf_seq2seq_sine', list(range(n_tests)), indirect=True)
def test_seq2seq(tf_seq2seq_sine):
    pass
