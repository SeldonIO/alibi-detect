from itertools import product
import numpy as np
import pytest
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from alibi_detect.ad import ModelDistillation
from alibi_detect.version import __version__

threshold = [None, 5.]
loss_type = ['kld', 'xent']
threshold_perc = [90.]
return_instance_score = [True, False]

tests = list(product(threshold, loss_type, threshold_perc, return_instance_score))
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
def adv_md_params(request):
    return tests[request.param]


@pytest.mark.parametrize('adv_md_params', list(range(n_tests)), indirect=True)
def test_adv_md(adv_md_params):
    # ModelDistillation parameters
    threshold, loss_type, threshold_perc, return_instance_score = adv_md_params

    # define ancillary model
    layers = [tf.keras.layers.InputLayer(input_shape=(input_dim)),
              tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax)]
    distilled_model = tf.keras.Sequential(layers)

    # init ModelDistillation detector
    admd = ModelDistillation(
        threshold=threshold,
        model=model,
        distilled_model=distilled_model,
        loss_type=loss_type
    )

    assert admd.threshold == threshold
    assert admd.meta == {'name': 'ModelDistillation', 'detector_type': 'offline', 'data_type': None,
                         'version': __version__}
    for layer in admd.model.layers:
        assert not layer.trainable

    # fit AdversarialVAE, infer threshold and compute scores
    admd.fit(X, epochs=5, verbose=False)
    admd.infer_threshold(X, threshold_perc=threshold_perc)
    iscore = admd.score(X)
    perc_score = 100 * (iscore < admd.threshold).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 1 > perc_score > threshold_perc - 1

    # make and check predictions
    ad_preds = admd.predict(X, return_instance_score=return_instance_score)

    assert ad_preds['meta'] == admd.meta
    if return_instance_score:
        assert ad_preds['data']['is_adversarial'].sum() == (ad_preds['data']['instance_score']
                                                            > admd.threshold).astype(int).sum()
    else:
        assert ad_preds['data']['instance_score'] is None
