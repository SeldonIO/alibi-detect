from itertools import product
import numpy as np
import pytest
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.od import OutlierVAE
from alibi_detect.models.tensorflow.losses import elbo
from alibi_detect.version import __version__

threshold = [None, 5.]
score_type = ['mse']
samples = [10]
loss_fn = [elbo, tf.keras.losses.mse]
threshold_perc = [90.]
return_instance_score = [True, False]
return_feature_score = [True, False]
outlier_perc = [50, 100]
outlier_type = ['instance', 'feature']

tests = list(product(threshold, score_type, samples, loss_fn, threshold_perc,
                     return_instance_score, return_feature_score, outlier_perc, outlier_type))
n_tests = len(tests)

# load iris data
X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)

input_dim = X.shape[1]
latent_dim = 2


@pytest.fixture
def vae_params(request):
    return tests[request.param]


@pytest.mark.parametrize('vae_params', list(range(n_tests)), indirect=True)
def test_vae(vae_params):
    # OutlierVAE parameters
    threshold, score_type, samples, loss_fn, threshold_perc, return_instance_score, \
        return_feature_score, outlier_perc, outlier_type = vae_params

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

    # init OutlierVAE
    vae = OutlierVAE(
        threshold=threshold,
        score_type=score_type,
        encoder_net=encoder_net,
        decoder_net=decoder_net,
        latent_dim=latent_dim,
        samples=samples
    )

    assert vae.threshold == threshold
    assert vae.meta == {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': None, 'version': __version__}

    # fit OutlierVAE, infer threshold and compute scores
    vae.fit(X, loss_fn=loss_fn, epochs=5, verbose=False)
    vae.infer_threshold(X, threshold_perc=threshold_perc)
    fscore, iscore = vae.score(X)
    perc_score = 100 * (iscore < vae.threshold).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vae.predict(X,
                           return_instance_score=return_instance_score,
                           return_feature_score=return_feature_score,
                           outlier_type=outlier_type,
                           outlier_perc=outlier_perc
                           )

    assert od_preds['meta'] == vae.meta
    if outlier_type == 'instance':
        assert od_preds['data']['is_outlier'].shape == (X.shape[0],)
        if return_instance_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score']
                                                            > vae.threshold).astype(int).sum()
    elif outlier_type == 'feature':
        assert od_preds['data']['is_outlier'].shape == X.shape
        if return_feature_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['feature_score']
                                                            > vae.threshold).astype(int).sum()

    if return_feature_score:
        assert od_preds['data']['feature_score'].shape == X.shape
    else:
        assert od_preds['data']['feature_score'] is None

    if return_instance_score:
        assert od_preds['data']['instance_score'].shape == (X.shape[0],)
    else:
        assert od_preds['data']['instance_score'] is None
