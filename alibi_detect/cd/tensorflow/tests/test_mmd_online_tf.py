from functools import partial
from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
from typing import Callable, List
from alibi_detect.cd.tensorflow.mmd_online import MMDDriftOnlineTF
from alibi_detect.cd.tensorflow.preprocess import HiddenOutput, UAE, preprocess_drift

n, n_hidden, n_classes = 400, 10, 5

tf.random.set_seed(0)


def mymodel(shape):
    x_in = Input(shape=shape)
    x = Dense(n_hidden)(x_in)
    x_out = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


def preprocess_list(x: List[np.ndarray]) -> np.ndarray:
    if len(x) > 1:  # test List[Any] reference data inputs to the detector with Any=np.ndarray
        return np.concatenate(x, axis=0)
    else:  # test Any inputs to the prediction function of the detector with Any=List[np.ndarray]
        return np.array(x)[0]


n_features = [10]
n_enc = [None, 3]
ert = [25]
window_size = [5]
preprocess = [
    (None, None),
    (preprocess_drift, {'model': HiddenOutput, 'layer': -1}),
    (preprocess_drift, {'model': UAE}),
    (preprocess_list, None)
]
n_bootstraps = [200]
tests_mmddriftonline = list(product(n_features, n_enc, ert, window_size, preprocess, n_bootstraps))
n_tests = len(tests_mmddriftonline)


@pytest.fixture
def mmd_online_params(request):
    return tests_mmddriftonline[request.param]


@pytest.mark.parametrize('mmd_online_params', list(range(n_tests)), indirect=True)
def test_mmd_online(mmd_online_params):
    n_features, n_enc, ert, window_size, preprocess, n_bootstraps = mmd_online_params

    np.random.seed(0)

    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    preprocess_fn, preprocess_kwargs = preprocess
    to_list = False
    if hasattr(preprocess_fn, '__name__') and preprocess_fn.__name__ == 'preprocess_list':
        to_list = True
        x_ref = [_[None, :] for _ in x_ref]
    elif isinstance(preprocess_fn, Callable):
        if 'layer' in list(preprocess_kwargs.keys()) \
                and preprocess_kwargs['model'].__name__ == 'HiddenOutput':
            model = mymodel((n_features,))
            layer = preprocess_kwargs['layer']
            preprocess_fn = partial(preprocess_fn, model=HiddenOutput(model=model, layer=layer))
        elif preprocess_kwargs['model'].__name__ == 'UAE' \
                and n_features > 1 and isinstance(n_enc, int):
            tf.random.set_seed(0)
            encoder_net = tf.keras.Sequential(
                [
                    InputLayer(input_shape=(n_features,)),
                    Dense(n_enc)
                ]
            )
            preprocess_fn = partial(preprocess_fn, model=UAE(encoder_net=encoder_net))
        else:
            preprocess_fn = None
    else:
        preprocess_fn = None

    cd = MMDDriftOnlineTF(
        x_ref=x_ref,
        ert=ert,
        window_size=window_size,
        preprocess_fn=preprocess_fn,
        n_bootstraps=n_bootstraps
    )

    x_h0 = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    detection_times_h0 = []
    test_stats_h0 = []
    for x_t in x_h0:
        if to_list:
            x_t = [x_t]
        pred_t = cd.predict(x_t, return_test_stat=True)
        test_stats_h0.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h0.append(pred_t['data']['time'])
            cd.reset()
    average_delay_h0 = np.array(detection_times_h0).mean()
    test_stats_h0 = [ts for ts in test_stats_h0 if ts is not None]
    assert ert/3 < average_delay_h0 < 3*ert

    cd.reset()

    x_h1 = 1 + np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    detection_times_h1 = []
    test_stats_h1 = []
    for x_t in x_h1:
        if to_list:
            x_t = [x_t]
        pred_t = cd.predict(x_t, return_test_stat=True)
        test_stats_h1.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h1.append(pred_t['data']['time'])
            cd.reset()
    average_delay_h1 = np.array(detection_times_h1).mean()
    test_stats_h1 = [ts for ts in test_stats_h1 if ts is not None]
    assert np.abs(average_delay_h1) < ert/2

    assert np.mean(test_stats_h1) > np.mean(test_stats_h0)
