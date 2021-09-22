from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.od import OutlierVAEGMM
from alibi_detect.version import __version__

threshold = [None, 5.]
n_gmm = [1, 2]
w_energy = [.1, .5]
w_recon = [0., 1e-7]
samples = [1, 10]
threshold_perc = [90.]
return_instance_score = [True, False]

tests = list(product(threshold, n_gmm, w_energy, w_recon, samples, threshold_perc, return_instance_score))
n_tests = len(tests)

# load and preprocess MNIST data
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X = X_train.reshape(X_train.shape[0], -1)[:1000]  # only train on 1000 instances
X = X.astype(np.float32)
X /= 255

input_dim = X.shape[1]
latent_dim = 2


@pytest.fixture
def vaegmm_params(request):
    return tests[request.param]


@pytest.mark.parametrize('vaegmm_params', list(range(n_tests)), indirect=True)
def test_vaegmm(vaegmm_params):
    # OutlierVAEGMM parameters
    threshold, n_gmm, w_energy, w_recon, samples, threshold_perc, return_instance_score = vaegmm_params

    # define encoder, decoder and GMM density net
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

    gmm_density_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim + 2,)),
            Dense(10, activation=tf.nn.relu),
            Dense(n_gmm, activation=tf.nn.softmax)
        ]
    )

    # init OutlierAEGMM
    vaegmm = OutlierVAEGMM(
        threshold=threshold,
        encoder_net=encoder_net,
        decoder_net=decoder_net,
        gmm_density_net=gmm_density_net,
        n_gmm=n_gmm,
        latent_dim=latent_dim,
        samples=samples
    )

    assert vaegmm.threshold == threshold
    assert vaegmm.meta == {'name': 'OutlierVAEGMM', 'detector_type': 'offline', 'data_type': None,
                           'version': __version__}

    # fit OutlierAEGMM, infer threshold and compute scores
    vaegmm.fit(X, w_recon=w_recon, w_energy=w_energy, epochs=5, batch_size=1000, verbose=False)
    vaegmm.infer_threshold(X, threshold_perc=threshold_perc)
    energy = vaegmm.score(X)
    perc_score = 100 * (energy < vaegmm.threshold).astype(int).sum() / energy.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vaegmm.predict(X, return_instance_score=return_instance_score)
    assert od_preds['meta'] == vaegmm.meta
    assert od_preds['data']['is_outlier'].shape == (X.shape[0],)
    if return_instance_score:
        assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score']
                                                        > vaegmm.threshold).astype(int).sum()
    else:
        assert od_preds['data']['instance_score'] is None
