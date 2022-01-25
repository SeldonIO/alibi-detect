"""
Tests for saving/loading of detectors via config.toml files.

Internal functions such as save_kernel/load_kernel etc are also tested.
"""
# TODO - test pytorch save/load functionality
from functools import partial
import numpy as np
import scipy
import pytest
# from pytest_lazyfixture import lazy_fixture
from sklearn.model_selection import StratifiedKFold
from tempfile import TemporaryDirectory
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Conv1D, Flatten
from typing import Callable
from alibi_detect.cd import (ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift, FETDrift,
                             LSDDDrift, SpotTheDiffDrift, LearnedKernelDrift)  # ), ClassifierUncertaintyDrift)
from packaging import version
if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift
from alibi_detect.cd.tensorflow import UAE, preprocess_drift
from alibi_detect.utils.tensorflow.kernels import DeepKernel
from alibi_detect.utils.saving import save_detector  # type: ignore
from alibi_detect.utils.loading import load_detector  # type: ignore

input_dim = 4
latent_dim = 2
n_gmm = 2
threshold = 10.
threshold_drift = .55
n_folds_drift = 5
samples = 6
seq_len = 10
p_val = .05
X_ref = np.random.rand(samples * input_dim).reshape(samples, input_dim)
X_ref_cat = np.tile(np.array([np.arange(samples)] * input_dim).T, (2, 1))
X_ref_mix = X_ref.copy()
X_ref_mix[:, 0] = np.tile(np.array(np.arange(samples // 2)), (1, 2)).T[:, 0]
X_ref_bin = np.random.choice([0, 1], (samples, input_dim), p=[0.6, 0.4])
n_permutations = 10

# define encoder and decoder
encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(input_dim,)),
        Dense(5, activation=tf.nn.relu),
        Dense(latent_dim, activation=None)
    ]
)

preprocess_fn = partial(preprocess_drift, model=UAE(encoder_net=encoder_net))

gmm_density_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim + 2,)),
        Dense(10, activation=tf.nn.relu),
        Dense(n_gmm, activation=tf.nn.softmax)
    ]
)

threshold_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(seq_len, latent_dim)),
        Dense(5, activation=tf.nn.relu)
    ]
)

# define classifier model
inputs = tf.keras.Input(shape=(input_dim,))
outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Deep kernel projection
proj = tf.keras.Sequential(
  [
      InputLayer((1, 1, input_dim,)),
      Conv1D(int(input_dim), 2, strides=1, padding='same', activation=tf.nn.relu),
      Conv1D(input_dim, 2, strides=1, padding='same', activation=tf.nn.relu),
      Flatten(),
  ]
)
deep_kernel = DeepKernel(proj, eps=0.01)

detector = [
    KSDrift(X_ref,
            p_val=p_val,
            preprocess_at_init=False,
            preprocess_fn=preprocess_fn),
    FETDrift(X_ref_bin,
             p_val=p_val,
             preprocess_at_init=True,
             alternative='less'),
    MMDDrift(X_ref,
             p_val=p_val,
             preprocess_at_init=False,
             preprocess_fn=preprocess_fn,
             configure_kernel_from_x_ref=True,
             n_permutations=n_permutations),
    LSDDDrift(X_ref,
              p_val=p_val,
              preprocess_at_init=False,
              preprocess_fn=preprocess_fn,
              n_permutations=n_permutations),
    ChiSquareDrift(X_ref_cat,
                   p_val=p_val,
                   preprocess_at_init=True),
    TabularDrift(X_ref_mix,
                 p_val=p_val,
                 categories_per_feature={0: None},
                 preprocess_at_init=True),
    ClassifierDrift(X_ref,
                    model=model,
                    p_val=p_val,
                    n_folds=n_folds_drift,
                    train_size=None,
                    preprocess_at_init=True),
    SpotTheDiffDrift(X_ref,
                     p_val=p_val,
                     n_folds=n_folds_drift,
                     train_size=None),
    LearnedKernelDrift(X_ref[:, None, :],
                       deep_kernel,
                       p_val=p_val,
                       train_size=0.7)
]
if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    detector.append(
        CVMDrift(X_ref,
                 p_val=p_val,
                 preprocess_at_init=True,
                 preprocess_fn=preprocess_fn)
    )
# TODO: ClassifierUncertaintyDrift
n_tests = len(detector)


@pytest.fixture
def select_detector(request):
    return detector[request.param]


@pytest.mark.parametrize('select_detector', list(range(n_tests)), indirect=True)
def test_save_load(select_detector):
    """
    Test of simple save/load functionality. Relatively simple detectors are instantiated, before being saved
    to a temporary directly and then loaded again. For deterministic detectors, the predicted p-values are compared
    to those predicted by the original detector.
    """
    det = select_detector
    det_name = det.meta['name']

    with TemporaryDirectory() as temp_dir:
        temp_dir += '/'
        save_detector(det, temp_dir)
        det_load = load_detector(temp_dir)
        det_load_name = det_load.meta['name']
        assert det_load_name == det_name

        if type(det_load) in [KSDrift, CVMDrift]:
            assert det_load.n_features == latent_dim
            assert det_load.p_val == p_val
            if type(det_load) == CVMDrift:
                x = preprocess_fn(X_ref)  # det_load.x_ref should be the preprocessed x_ref after save/load
                assert det_load.x_ref_preprocessed
            else:
                x = X_ref
            np.testing.assert_array_equal(det_load.x_ref, x)
            assert isinstance(det_load.preprocess_fn, Callable)
            assert det_load.preprocess_fn.func.__name__ == 'preprocess_drift'
            np.testing.assert_array_equal(det.predict(X_ref)['data']['p_val'],  # only do for deterministic detectors
                                          det_load.predict(X_ref)['data']['p_val'])
        elif type(det_load) == FETDrift:
            assert det_load.n_features == input_dim
            np.testing.assert_array_equal(det_load.x_ref, X_ref_bin)
            print(det.predict(X_ref_bin))
            np.testing.assert_array_equal(det.predict(X_ref_bin)['data']['p_val'],
                                          det_load.predict(X_ref_bin)['data']['p_val'])
        elif type(det_load) in [ChiSquareDrift, TabularDrift]:
            assert isinstance(det_load.x_ref_categories, dict)
            assert det_load.p_val == p_val
            x = X_ref_cat.copy() if isinstance(det_load, ChiSquareDrift) else X_ref_mix.copy()
            np.testing.assert_array_equal(det_load.x_ref, x)
            np.testing.assert_array_equal(det.predict(x)['data']['p_val'],
                                          det_load.predict(x)['data']['p_val'])
        elif type(det_load) == MMDDrift:
            assert not det_load._detector.infer_sigma
            assert det_load._detector.n_permutations == n_permutations
            assert det_load._detector.p_val == p_val
            np.testing.assert_array_equal(det_load._detector.x_ref, X_ref)
            assert isinstance(det_load._detector.preprocess_fn, Callable)
            assert det_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
            # assert det.predict(X_ref)['data']['p_val'] == det_load.predict(X_ref)['data']['p_val']
            # Commented as settings tf/np seeds does not currently make deterministic
        elif type(det_load) == LSDDDrift:
            assert det_load._detector.n_permutations == n_permutations
            assert det_load._detector.p_val == p_val
            np.testing.assert_array_equal(det_load._detector.x_ref, X_ref)
            assert isinstance(det_load._detector.preprocess_fn, Callable)
            assert det_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
            # assert det.predict(X_ref)['data']['p_val'] == det_load.predict(X_ref)['data']['p_val']
            # Commented as settings tf/np seeds does not currently make deterministic
        elif type(det_load) == ClassifierDrift:
            assert det_load._detector.p_val == p_val
            np.testing.assert_array_equal(det_load._detector.x_ref, X_ref)
            assert isinstance(det_load._detector.skf, StratifiedKFold)
            assert isinstance(det_load._detector.train_kwargs, dict)
            assert isinstance(det_load._detector.model, tf.keras.Model)
        elif type(det_load) == SpotTheDiffDrift:
            assert det_load._detector._detector.p_val == p_val
            np.testing.assert_array_equal(det_load._detector._detector.x_ref, X_ref)
            assert isinstance(det_load._detector._detector.skf, StratifiedKFold)
            assert isinstance(det_load._detector._detector.train_kwargs, dict)
            assert isinstance(det_load._detector._detector.model, tf.keras.Model)
        elif type(det_load) == LearnedKernelDrift:
            x = X_ref[:, None, :]
            assert det_load._detector.p_val == p_val
            np.testing.assert_array_equal(det_load._detector.x_ref, x)
            assert isinstance(det_load._detector.train_kwargs, dict)
            assert isinstance(det_load._detector.kernel, DeepKernel)

        # TODO - checks for modeluncertainty

# TODO- unit tests
#       - save/load_kernel
#       - save/load_tokenizer
#       - save/load_embedding
#       - save/load_model
#       - serialize_preprocess/load_preprocessor
#       - serialize_function
#       - _resolve_paths
#       - load_optimizer
#       - validate_config
#       - _load_detector_config
#       - init_detector
#       - prep_model_and_embedding
#       - get_nested_value
#       - set_nested_value
#       - read_config
#       - resolve_cfg
#       - set_device
#       - _replace
#       - save/load_tf_model
#       - registry!
