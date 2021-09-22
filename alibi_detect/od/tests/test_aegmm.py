from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.od import OutlierAEGMM
from alibi_detect.version import __version__

threshold = [None, 5.]
n_gmm = [1, 2]
w_energy = [.1, .5]
threshold_perc = [90.]
return_instance_score = [True, False]

tests = list(product(threshold, n_gmm, w_energy, threshold_perc, return_instance_score))
n_tests = len(tests)

# load and preprocess MNIST data
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X = X_train.reshape(X_train.shape[0], -1)[:1000]  # only train on 1000 instances
X = X.astype(np.float32)
X /= 255

input_dim = X.shape[1]
latent_dim = 2


@pytest.fixture
def aegmm_params(request):
    return tests[request.param]


@pytest.mark.parametrize('aegmm_params', list(range(n_tests)), indirect=True)
def test_aegmm(aegmm_params):
    # OutlierAEGMM parameters
    threshold, n_gmm, w_energy, threshold_perc, return_instance_score = aegmm_params

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
    aegmm = OutlierAEGMM(
        threshold=threshold,
        encoder_net=encoder_net,
        decoder_net=decoder_net,
        gmm_density_net=gmm_density_net,
        n_gmm=n_gmm
    )

    assert aegmm.threshold == threshold
    assert aegmm.meta == {'name': 'OutlierAEGMM', 'detector_type': 'offline', 'data_type': None, 'version': __version__}

    # fit OutlierAEGMM, infer threshold and compute scores
    aegmm.fit(X, w_energy=w_energy, epochs=5, batch_size=1000, verbose=False)
    aegmm.infer_threshold(X, threshold_perc=threshold_perc)
    energy = aegmm.score(X)
    perc_score = 100 * (energy < aegmm.threshold).astype(int).sum() / energy.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = aegmm.predict(X, return_instance_score=return_instance_score)
    assert od_preds['meta'] == aegmm.meta
    assert od_preds['data']['is_outlier'].shape == (X.shape[0],)
    if return_instance_score:
        assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score']
                                                        > aegmm.threshold).astype(int).sum()
    else:
        assert od_preds['data']['instance_score'] is None
