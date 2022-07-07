from functools import partial
from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer
from typing import Callable, List
from alibi_detect.cd.tensorflow.lsdd import LSDDDriftTF
from alibi_detect.cd.tensorflow.preprocess import HiddenOutput, UAE, preprocess_drift

n, n_hidden, n_classes = 500, 10, 5

tf.random.set_seed(0)


def mymodel(shape):
    x_in = Input(shape=shape)
    x = Dense(n_hidden)(x_in)
    x_out = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


# test List[Any] inputs to the detector
def preprocess_list(x: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(x, axis=0)


n_features = [10]
n_enc = [None, 3]
preprocess = [
    (None, None),
    (preprocess_drift, {'model': HiddenOutput, 'layer': -1}),
    (preprocess_drift, {'model': UAE}),
    (preprocess_list, None)
]
update_x_ref = [None]
preprocess_x_ref = [True, False]
n_permutations = [10]
tests_lsdddrift = list(product(n_features, n_enc, preprocess,
                               n_permutations, update_x_ref, preprocess_x_ref))
n_tests = len(tests_lsdddrift)


@pytest.fixture
def lsdd_params(request):
    return tests_lsdddrift[request.param]


@pytest.mark.parametrize('lsdd_params', list(range(n_tests)), indirect=True)
def test_lsdd(lsdd_params):
    n_features, n_enc, preprocess, n_permutations, update_x_ref, preprocess_x_ref = lsdd_params

    np.random.seed(0)

    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    preprocess_fn, preprocess_kwargs = preprocess
    to_list = False
    if hasattr(preprocess_fn, '__name__') and preprocess_fn.__name__ == 'preprocess_list':
        if not preprocess_x_ref:
            return
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

    cd = LSDDDriftTF(
        x_ref=x_ref,
        p_val=.05,
        preprocess_x_ref=preprocess_x_ref if isinstance(preprocess_fn, Callable) else False,
        update_x_ref=update_x_ref,
        preprocess_fn=preprocess_fn,
        n_permutations=n_permutations
    )

    perturbation = np.random.normal(size=(n, n_features)) / 100  # LSDD struggles with copies/repeats
    x = x_ref.copy() + perturbation.astype(np.float32)
    preds = cd.predict(x, return_p_val=True)
    assert preds['data']['is_drift'] == 0 and preds['data']['p_val'] >= cd.p_val
    if isinstance(update_x_ref, dict):
        k = list(update_x_ref.keys())[0]
        assert cd.n == len(x) + len(x_ref)
        assert cd.x_ref.shape[0] == min(update_x_ref[k], len(x) + len(x_ref))

    x_h1 = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    if to_list:
        x_h1 = [_[None, :] for _ in x_h1]
    preds = cd.predict(x_h1, return_p_val=True)
    if preds['data']['is_drift'] == 1:
        assert preds['data']['p_val'] < preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] > preds['data']['distance_threshold']
    else:
        assert preds['data']['p_val'] >= preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] <= preds['data']['distance_threshold']
