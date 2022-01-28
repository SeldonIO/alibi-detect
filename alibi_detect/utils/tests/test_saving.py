"""
Tests for saving/loading of detectors via config.toml files.

Internal functions such as save_kernel/load_kernel etc are also tested.
"""
# TODO - test pytorch save/load functionality
from functools import partial
import numpy as np
import scipy
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture, param_fixture
from sklearn.model_selection import StratifiedKFold
from tempfile import TemporaryDirectory
import tensorflow as tf
from typing import Callable
from packaging import version
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_pt
from alibi_detect.utils.tensorflow.kernels import DeepKernel
import torch
from alibi_detect.utils.saving import save_detector  # type: ignore
from alibi_detect.utils.loading import load_detector  # type: ignore
from alibi_detect.cd import (
    ChiSquareDrift,
    ClassifierDrift,
    KSDrift,
    MMDDrift,
    TabularDrift,
    FETDrift,
    LSDDDrift,
    SpotTheDiffDrift,
    LearnedKernelDrift,
    #  ClassifierUncertaintyDrift,
)
from datasets import ContinuousData, BinData, CategoricalData, MixedData
if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift

backend = param_fixture("backend", ['tensorflow'])
P_VAL = 0.05
N_PERMUTATIONS = 10
LATENT_DIM = 2  # Must be less than input_dim set in ./datasets.py


@fixture  # If wanted to test more than one preprocess_fn, could parametrize this
def preprocess_uae(backend, current_cases):
    """
    Preprocess function with Untrained Autoencoder of given input dimension and backend.
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']

    if backend == 'tensorflow':
        encoder_net = tf.keras.Sequential(
               [
                   tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                   tf.keras.layers.Dense(5, activation=tf.nn.relu),
                   tf.keras.layers.Dense(LATENT_DIM, activation=None)
               ]
           )
        preprocess_fn = partial(preprocess_drift_tf, model=encoder_net)
    elif backend == 'pytorch':
        raise NotImplementedError('`pytorch` tests not implemented.')
    else:
        raise ValueError('preprocess_uae `backend` not valid.')

    return preprocess_fn


@fixture
def deep_kernel(backend, current_cases):
    """
    Deep kernel with given input function and backend.
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']

    if backend == 'tensorflow':
        proj = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer((1, 1, input_dim,)),
              tf.keras.layers.Conv1D(int(input_dim), 2, strides=1, padding='same', activation=tf.nn.relu),
              tf.keras.layers.Conv1D(input_dim, 2, strides=1, padding='same', activation=tf.nn.relu),
              tf.keras.layers.Flatten(),
          ]
        )
        deep_kernel = DeepKernel(proj, eps=0.01)
    elif backend == 'pytorch':
        raise NotImplementedError('`pytorch` tests not implemented.')
    else:
        raise ValueError('preprocess_uae `backend` not valid.')
    return deep_kernel


@fixture
def classifier(backend, current_cases):
    """
    Classification model with given input function and backend.
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']
    if backend == 'tensorflow':
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    elif backend == 'pytorch':
        raise NotImplementedError('`pytorch` tests not implemented.')
    else:
        raise ValueError('preprocess_uae `backend` not valid.')
    return model


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_ksdrift(data, preprocess_uae, tmp_path):
    """ Test KSDrift on continuous datasets, with UAE as preprocess_fn."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = KSDrift(X_ref,
                 p_val=P_VAL,
                 preprocess_fn=preprocess_uae,
                 preprocess_at_init=True,
                 )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load.x_ref)
    assert cd_load.x_ref_preprocessed
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@pytest.mark.skipif(version.parse(scipy.__version__) < version.parse('1.7.0'),
                    reason="Requires scipy version >= 1.7.0")
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_cvmdrift(data, preprocess_uae, tmp_path):
    """ Test CVMDrift on continuous datasets, with UAE as preprocess_fn."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = CVMDrift(X_ref,
                 p_val=P_VAL,
                 preprocess_fn=preprocess_uae,
                 preprocess_at_init=True,
                 )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load.x_ref)
    assert cd_load.x_ref_preprocessed
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_mmddrift(data, preprocess_uae, backend, tmp_path):
    """ Test MMDDrift on continuous datasets, with UAE as preprocess_fn."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = MMDDrift(X_ref,
                  p_val=P_VAL,
                  backend=backend,
                  preprocess_fn=preprocess_uae,
                  n_permutations=N_PERMUTATIONS,
                  preprocess_at_init=True,
                  )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # assertions
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.x_ref_preprocessed
    assert not cd_load._detector.infer_sigma
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    # assert det.predict(X_ref)['data']['p_val'] == det_load.predict(X_ref)['data']['p_val']
    # Commented as settings tf/np seeds does not currently make deterministic


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_lsdddrift(data, preprocess_uae, backend, tmp_path):
    """ Test LSDDDrift on continuous datasets, with UAE as preprocess_fn."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = LSDDDrift(X_ref,
                   p_val=P_VAL,
                   backend=backend,
                   preprocess_fn=preprocess_uae,
                   n_permutations=N_PERMUTATIONS,
                   preprocess_at_init=True,
                   )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # assertions
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    # assert det.predict(X_ref)['data']['p_val'] == det_load.predict(X_ref)['data']['p_val']
    # Commented as settings tf/np seeds does not currently make deterministic


@parametrize_with_cases("data", cases=BinData, prefix='data_')
def test_save_fetdrift(data, tmp_path):
    """ Test FETDrift on binary datasets."""
    # Detector save/load
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cd = FETDrift(X_ref,
                  p_val=P_VAL,
                  alternative='less',
                  )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load.x_ref)
    assert not cd_load.x_ref_preprocessed
    assert cd_load.n_features == input_dim
    assert cd_load.p_val == P_VAL
    assert cd_load.alternative == 'less'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=CategoricalData, prefix='data_')
def test_save_chisquaredrift(data, tmp_path):
    """ Test ChiSquareDrift on categorical datasets."""
    # Detector save/load
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cd = ChiSquareDrift(X_ref,
                        p_val=P_VAL,
                        )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load.x_ref)
    assert not cd_load.x_ref_preprocessed
    assert cd_load.n_features == input_dim
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.x_ref_categories, dict)
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=MixedData, prefix='data_')
def test_save_tabulardrift(data, tmp_path):
    """ Test TabularDrift on mixed datasets."""
    # Detector save/load
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cd = TabularDrift(X_ref,
                      p_val=P_VAL,
                      categories_per_feature={0: None},
                      )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load.x_ref)
    assert not cd_load.x_ref_preprocessed
    assert cd_load.n_features == input_dim
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.x_ref_categories, dict)
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_classifierdrift(data, classifier, backend, tmp_path):
    """ Test ClassifierDrift on continuous datasets."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = ClassifierDrift(X_ref,
                         model=classifier,
                         p_val=P_VAL,
                         n_folds=5,
                         backend=backend,
                         train_size=None)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load._detector.x_ref)
    assert isinstance(cd_load._detector.skf, StratifiedKFold)
    assert not cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.model, tf.keras.Model)
    else:
        pass  # TODO
#    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
#                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_spotthediff(data, classifier, preprocess_uae, backend, tmp_path):
    """ Test SpotTheDiffDrift on continuous datasets, with UAE as preprocess_fn."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = SpotTheDiffDrift(X_ref,
                          p_val=P_VAL,
                          n_folds=5,
                          train_size=None,
                          backend=backend,
                          preprocess_fn=preprocess_uae)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load._detector._detector.x_ref)
    assert isinstance(cd_load._detector._detector.skf, StratifiedKFold)
    assert cd_load._detector._detector.x_ref_preprocessed
    assert cd_load._detector._detector.p_val == P_VAL
    assert isinstance(cd_load._detector._detector.preprocess_fn, Callable)
    assert cd_load._detector._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert isinstance(cd_load._detector._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector._detector.model, tf.keras.Model)
    else:
        pass  # TODO


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_learnedkernel(data, deep_kernel, preprocess_uae, backend, tmp_path):
    """ Test LearnedKernelDrift on continuous datasets, with UAE as preprocess_fn."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = LearnedKernelDrift(X_ref[:, None, :],
                            deep_kernel,
                            p_val=P_VAL,
                            backend=backend,
                            preprocess_fn=preprocess_uae,
                            train_size=0.7)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    x = X_ref[:, None, :]
    np.testing.assert_array_equal(preprocess_uae(x), cd_load._detector.x_ref)
    assert cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.train_kwargs, dict)
    assert isinstance(cd_load._detector.kernel, DeepKernel)


# TODO - checks for modeluncertainty detectors
#
## TODO- unit tests
##       - save/load_kernel
##       - save/load_tokenizer
##       - save/load_embedding
##       - save/load_model
##       - serialize_preprocess/load_preprocessor
##       - serialize_function
##       - _resolve_paths
##       - load_optimizer
##       - validate_config
##       - _load_detector_config
##       - init_detector
##       - prep_model_and_embedding
##       - get_nested_value
##       - set_nested_value
##       - read_config
##       - resolve_cfg
##       - set_device
##       - _replace
##       - save/load_tf_model
##       - registry! - Do in separate file
#