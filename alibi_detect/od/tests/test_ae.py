from itertools import product
import numpy as np
import pytest
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.od import OutlierAE
from alibi_detect.version import __version__

threshold = [None, 5.]
threshold_perc = [90.]
return_instance_score = [True, False]
return_feature_score = [True, False]
outlier_perc = [50, 100]
outlier_type = ['instance', 'feature']

tests = list(product(threshold, threshold_perc, return_instance_score,
                     return_feature_score, outlier_perc, outlier_type))
n_tests = len(tests)

# load iris data
X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)

input_dim = X.shape[1]
encoding_dim = 1


@pytest.fixture
def ae_params(request):
    return tests[request.param]


@pytest.mark.parametrize('ae_params', list(range(n_tests)), indirect=True)
def test_ae(ae_params):
    # OutlierAE parameters
    threshold, threshold_perc, return_instance_score, return_feature_score, outlier_perc, outlier_type = ae_params

    # define encoder and decoder
    encoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(input_dim,)),
            Dense(5, activation=tf.nn.relu),
            Dense(encoding_dim, activation=None)
        ]
    )

    decoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(encoding_dim,)),
            Dense(5, activation=tf.nn.relu),
            Dense(input_dim, activation=tf.nn.sigmoid)
        ]
    )

    # init OutlierAE
    ae = OutlierAE(
        threshold=threshold,
        encoder_net=encoder_net,
        decoder_net=decoder_net
    )

    assert ae.threshold == threshold
    assert ae.meta == {'name': 'OutlierAE', 'detector_type': 'offline', 'data_type': None, 'version': __version__}

    # fit OutlierAE, infer threshold and compute scores
    ae.fit(X, epochs=5, verbose=False)
    ae.infer_threshold(X, threshold_perc=threshold_perc)
    fscore, iscore = ae.score(X)
    perc_score = 100 * (iscore < ae.threshold).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = ae.predict(X,
                          return_instance_score=return_instance_score,
                          return_feature_score=return_feature_score,
                          outlier_type=outlier_type,
                          outlier_perc=outlier_perc
                          )

    assert od_preds['meta'] == ae.meta
    if outlier_type == 'instance':
        assert od_preds['data']['is_outlier'].shape == (X.shape[0],)
        if return_instance_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score']
                                                            > ae.threshold).astype(int).sum()
    elif outlier_type == 'feature':
        assert od_preds['data']['is_outlier'].shape == X.shape
        if return_feature_score:
            assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['feature_score']
                                                            > ae.threshold).astype(int).sum()

    if return_feature_score:
        assert od_preds['data']['feature_score'].shape == X.shape
    else:
        assert od_preds['data']['feature_score'] is None

    if return_instance_score:
        assert od_preds['data']['instance_score'].shape == (X.shape[0],)
    else:
        assert od_preds['data']['instance_score'] is None
