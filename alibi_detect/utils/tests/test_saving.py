import pytest
from tempfile import TemporaryDirectory
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.ad import AdversarialVAE
from alibi_detect.od import IForest, Mahalanobis, OutlierAEGMM, OutlierVAE, OutlierVAEGMM
from alibi_detect.utils.saving import save_detector, load_detector

input_dim = 4
latent_dim = 2
n_gmm = 2
threshold = 10.
samples = 5

# define encoder and decoder
encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(input_dim,)),
        Dense(5, activation=tf.nn.relu),
        Dense(latent_dim, activation=None)
    ]
)

decoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim,)),
        Dense(5, activation=tf.nn.relu),
        Dense(input_dim, activation=tf.nn.sigmoid)
    ]
)

kwargs = {'encoder_net': encoder_net,
          'decoder_net': decoder_net}

gmm_density_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim + 2,)),
        Dense(10, activation=tf.nn.relu),
        Dense(n_gmm, activation=tf.nn.softmax)
    ]
)

# define model
inputs = tf.keras.Input(shape=(input_dim,))
outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

detector = [
    AdversarialVAE(threshold=threshold,
                   model=model,
                   latent_dim=latent_dim,
                   samples=samples,
                   **kwargs),
    IForest(threshold=threshold),
    Mahalanobis(threshold=threshold),
    OutlierAEGMM(threshold=threshold,
                 gmm_density_net=gmm_density_net,
                 n_gmm=n_gmm,
                 **kwargs),
    OutlierVAE(threshold=threshold,
               latent_dim=latent_dim,
               samples=samples,
               **kwargs),
    OutlierVAEGMM(threshold=threshold,
                  gmm_density_net=gmm_density_net,
                  n_gmm=n_gmm,
                  latent_dim=latent_dim,
                  samples=samples,
                  **kwargs)
]
n_tests = len(detector)


@pytest.fixture
def select_detector(request):
    return detector[request.param]


@pytest.mark.parametrize('select_detector', list(range(n_tests)), indirect=True)
def test_save_load(select_detector):
    det = select_detector
    det_name = det.meta['name']

    with TemporaryDirectory() as temp_dir:
        temp_dir += '/'
        save_detector(det, temp_dir)
        det_load = load_detector(temp_dir)
        det_load_name = det_load.meta['name']
        assert det_load_name == det_name
        assert det_load.threshold == det.threshold == threshold

        if type(det_load) in [AdversarialVAE, OutlierVAE, OutlierVAEGMM]:
            assert det_load.samples == det.samples == samples

        if type(det_load) == AdversarialVAE:
            for layer in det_load.model.layers:
                assert not layer.trainable

        if type(det_load) == OutlierAEGMM:
            assert isinstance(det_load.aegmm.encoder, tf.keras.Sequential)
            assert isinstance(det_load.aegmm.decoder, tf.keras.Sequential)
            assert isinstance(det_load.aegmm.gmm_density, tf.keras.Sequential)
            assert isinstance(det_load.aegmm, tf.keras.Model)
            assert det_load.aegmm.n_gmm == n_gmm
        elif type(det_load) == OutlierVAEGMM:
            assert isinstance(det_load.vaegmm.encoder.encoder_net, tf.keras.Sequential)
            assert isinstance(det_load.vaegmm.decoder, tf.keras.Sequential)
            assert isinstance(det_load.vaegmm.gmm_density, tf.keras.Sequential)
            assert isinstance(det_load.vaegmm, tf.keras.Model)
            assert det_load.vaegmm.latent_dim == latent_dim
            assert det_load.vaegmm.n_gmm == n_gmm
        elif type(det_load) in [AdversarialVAE, OutlierVAE]:
            assert isinstance(det_load.vae.encoder.encoder_net, tf.keras.Sequential)
            assert isinstance(det_load.vae.decoder.decoder_net, tf.keras.Sequential)
            assert isinstance(det_load.vae, tf.keras.Model)
            assert det_load.vae.latent_dim == latent_dim
        elif type(det_load) == Mahalanobis:
            assert det_load.clip is None
            assert det_load.mean == det_load.C == det_load.n == 0
            assert det_load.meta['detector_type'] == 'online'
