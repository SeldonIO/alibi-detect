from itertools import product
import numpy as np
import pytest
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from alibi_detect.ad import AdversarialAE
from alibi_detect.version import __version__

threshold = [None, 5.]
w_model = [1., .5]
w_recon = [0., 1e-5]
threshold_perc = [90.]
return_instance_score = [True, False]

tests = list(product(threshold, w_model, w_recon, threshold_perc, return_instance_score))
n_tests = len(tests)

# load iris data
X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)
y = to_categorical(y)

input_dim = X.shape[1]
latent_dim = 2

# define and train model
inputs = tf.keras.Input(shape=(input_dim,))
outputs = tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, batch_size=150, epochs=10)


@pytest.fixture
def adv_ae_params(request):
    return tests[request.param]


@pytest.mark.parametrize('adv_ae_params', list(range(n_tests)), indirect=True)
def test_adv_vae(adv_ae_params):
    # AdversarialAE parameters
    threshold, w_model, w_recon, threshold_perc, return_instance_score = adv_ae_params

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
    advae = AdversarialAE(
        threshold=threshold,
        model=model,
        encoder_net=encoder_net,
        decoder_net=decoder_net
    )

    assert advae.threshold == threshold
    assert advae.meta == {'name': 'AdversarialAE', 'detector_type': 'offline', 'data_type': None,
                          'version': __version__}
    for layer in advae.model.layers:
        assert not layer.trainable

    # fit AdversarialVAE, infer threshold and compute scores
    advae.fit(X, w_model=w_model, w_recon=w_recon, epochs=5, verbose=False)
    advae.infer_threshold(X, threshold_perc=threshold_perc)
    iscore = advae.score(X)
    perc_score = 100 * (iscore < advae.threshold).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 1 > perc_score > threshold_perc - 1

    # make and check predictions
    ad_preds = advae.predict(X, return_instance_score=return_instance_score)

    assert ad_preds['meta'] == advae.meta
    if return_instance_score:
        assert ad_preds['data']['is_adversarial'].sum() == (ad_preds['data']['instance_score']
                                                            > advae.threshold).astype(int).sum()
    else:
        assert ad_preds['data']['instance_score'] is None
