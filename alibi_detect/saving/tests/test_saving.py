# type: ignore
"""
Tests for saving/loading of detectors via config.toml files.

Internal functions such as save_kernel/load_kernel_config etc are also tested.
"""
from functools import partial
import os
from pathlib import Path
from typing import Callable

import sklearn.base
import toml
import dill
import numpy as np
import pytest
import scipy
import tensorflow as tf
import torch
import torch.nn as nn

from .datasets import BinData, CategoricalData, ContinuousData, MixedData, TextData
from .models import (encoder_model, preprocess_uae, preprocess_hiddenoutput, preprocess_simple,  # noqa: F401
                     preprocess_simple_with_kwargs,
                     preprocess_nlp, LATENT_DIM, classifier_model, kernel, deep_kernel, nlp_embedding_and_tokenizer,
                     embedding, tokenizer, max_len, enc_dim, encoder_dropout_model, optimizer)

from alibi_detect.utils._random import fixed_seed
from packaging import version
from pytest_cases import param_fixture, parametrize, parametrize_with_cases
from sklearn.model_selection import StratifiedKFold


from alibi_detect.cd import (ChiSquareDrift, ClassifierUncertaintyDrift, RegressorUncertaintyDrift,
                             ClassifierDrift, FETDrift, KSDrift, LearnedKernelDrift, LSDDDrift, MMDDrift,
                             SpotTheDiffDrift, TabularDrift, ContextMMDDrift, MMDDriftOnline, LSDDDriftOnline,
                             CVMDriftOnline, FETDriftOnline)
from alibi_detect.models.pytorch import TransformerEmbedding as TransformerEmbedding_pt
from alibi_detect.models.tensorflow import TransformerEmbedding as TransformerEmbedding_tf
from alibi_detect.saving import (load_detector, read_config, registry,
                                 resolve_config, save_detector, write_config)
from alibi_detect.saving.loading import (_get_nested_value, _replace,
                                         _set_dtypes, _set_nested_value, _prepend_cfg_filepaths)
from alibi_detect.saving.saving import _serialize_object
from alibi_detect.saving.saving import (_path2str, _int2str_keys, _save_kernel_config, _save_model_config,
                                        _save_preprocess_config)
from alibi_detect.saving.schemas import DeepKernelConfig, KernelConfig, ModelConfig, PreprocessConfig
from alibi_detect.utils.pytorch.kernels import DeepKernel as DeepKernel_pt
from alibi_detect.utils.tensorflow.kernels import DeepKernel as DeepKernel_tf
from alibi_detect.utils.frameworks import has_keops
if has_keops:  # pykeops only installed in Linux CI
    from pykeops.torch import LazyTensor
    from alibi_detect.utils.keops.kernels import DeepKernel as DeepKernel_ke

if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift

# TODO: We currently parametrize encoder_model etc (in models.py) with backend, so the same flavour of
# preprocessing is used as the detector backend. In the future we could decouple this in tests.
backends = ['tensorflow', 'pytorch', 'sklearn']
if has_keops:  # pykeops only installed in Linux CI
    backends.append('keops')
backend = param_fixture("backend", backends)
P_VAL = 0.05
ERT = 10
N_PERMUTATIONS = 10
N_BOOTSTRAPS = 100
WINDOW_SIZE = 5
REGISTERED_OBJECTS = registry.get_all()

# Define a detector config dict
MMD_CFG = {
    'name': 'MMDDrift',
    'x_ref': np.array([[-0.30074928], [1.50240758], [0.43135768], [2.11295779], [0.79684913]]),
    'p_val': 0.05,
    'n_permutations': 150,
    'data_type': 'tabular'
}
CFGS = [MMD_CFG]

# TODO - future: Some of the fixtures can/should be moved elsewhere (i.e. if they can be recycled for use elsewhere)


@parametrize('cfg', CFGS)
def test_load_simple_config(cfg, tmp_path):
    """
    Test that a bare-bones `config.toml` without a [meta] field can be loaded by `load_detector`.
    """
    save_dir = tmp_path
    x_ref_path = str(save_dir.joinpath('x_ref.npy'))
    cfg_path = save_dir.joinpath('config.toml')
    # Save x_ref in config.toml
    x_ref = cfg['x_ref']
    np.save(x_ref_path, x_ref)
    cfg['x_ref'] = 'x_ref.npy'
    # Save config.toml then load it
    with open(cfg_path, 'w') as f:
        toml.dump(cfg, f)
    cd = load_detector(cfg_path)
    assert cd.__class__.__name__ == cfg['name']
    # Get config and compare to original (orginal cfg not fully spec'd so only compare items that are present)
    cfg_new = cd.get_config()
    for k, v in cfg.items():
        if k == 'x_ref':
            assert v == 'x_ref.npy'
        else:
            assert v == cfg_new[k]


@parametrize('preprocess_fn', [preprocess_uae, preprocess_hiddenoutput])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_ksdrift(data, preprocess_fn, tmp_path):
    """
    Test KSDrift on continuous datasets, with UAE and classifier_model softmax output as preprocess_fn's. Only this
    detector is tested with preprocessing strategies, as other detectors should see the same preprocess_fn output.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    cd = KSDrift(X_ref,
                 p_val=P_VAL,
                 preprocess_fn=preprocess_fn,
                 preprocess_at_init=True,
                 )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_fn(X_ref), cd_load.x_ref)
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],
                                  cd_load.predict(X_h0)['data']['p_val'])


@pytest.mark.skipif(backend == 'sklearn', reason="Don't test with sklearn preprocessing.")
@parametrize('preprocess_fn', [preprocess_nlp])
@parametrize_with_cases("data", cases=TextData.movie_sentiment_data, prefix='data_')
def test_save_ksdrift_nlp(data, preprocess_fn, enc_dim, tmp_path):  # noqa: F811
    """
    Test KSDrift on continuous datasets, with UAE and classifier_model softmax output as preprocess_fn's. Only this
    detector is tested with embedding and embedding+uae, as other detectors should see the same preprocessed data.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data['X_train'][:5], data['X_test'][:5]
    cd = KSDrift(X_ref,
                 p_val=P_VAL,
                 preprocess_fn=preprocess_fn,
                 preprocess_at_init=True,
                 input_shape=(768,),  # hardcoded to bert-base-cased for now
                 )
    save_detector(cd, tmp_path, legacy=False)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_fn(X_ref), cd_load.x_ref)
    if isinstance(preprocess_fn.keywords['model'], (TransformerEmbedding_tf, TransformerEmbedding_pt)):
        assert cd_load.n_features == 768  # hardcoded to bert-base-cased for now
    else:
        assert cd_load.n_features == enc_dim  # encoder dim
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],
                                  cd_load.predict(X_h0)['data']['p_val'])


@pytest.mark.skipif(version.parse(scipy.__version__) < version.parse('1.7.0'),
                    reason="Requires scipy version >= 1.7.0")
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_cvmdrift(data, preprocess_uae, tmp_path):
    """
    Test CVMDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize('kernel', [
        None,  # Use default kernel
        {'sigma': 0.5, 'trainable': False},  # pass kernel as object
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_mmddrift(data, kernel, preprocess_uae, backend, tmp_path, seed):  # noqa: F811
    """
    Test MMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch', 'keops'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and make predictions
    X_ref, X_h0 = data
    kwargs = {
        'p_val': P_VAL,
        'backend': backend,
        'preprocess_fn': preprocess_uae,
        'n_permutations': N_PERMUTATIONS,
        'preprocess_at_init': True,
        'kernel': kernel,
        'configure_kernel_from_x_ref': False,
        'sigma': np.array([0.5], dtype=np.float32)
    }
    if backend in ('pytorch', 'keops'):
        kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    with fixed_seed(seed):
        cd = MMDDrift(X_ref, **kwargs)
        preds = cd.predict(X_h0)
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)

    # assertions
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load._detector.x_ref)
    assert not cd_load._detector.infer_sigma
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert cd._detector.kernel.sigma == cd_load._detector.kernel.sigma
    assert cd._detector.kernel.init_sigma_fn == cd_load._detector.kernel.init_sigma_fn
    assert preds['data']['p_val'] == preds_load['data']['p_val']


# @parametrize('preprocess_fn', [preprocess_uae, preprocess_hiddenoutput])
@parametrize('preprocess_at_init', [True, False])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_lsdddrift(data, preprocess_at_init, backend, tmp_path, seed):
    """
    Test LSDDDrift on continuous datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    preprocess_fn = preprocess_simple
    # TODO - TensorFlow based preprocessors currently cause un-deterministic behaviour with LSDD permutations. Replace
    #  preprocess_simple with parametrized preprocess_fn's once above issue resolved.

    # Init detector and make predictions
    X_ref, X_h0 = data
    with fixed_seed(seed):  # Init and predict with a fixed random state
        cd = LSDDDrift(X_ref,
                       p_val=P_VAL,
                       backend=backend,
                       preprocess_fn=preprocess_fn,
                       preprocess_at_init=preprocess_at_init,
                       n_permutations=N_PERMUTATIONS
                       )
        preds = cd.predict(X_h0)
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):  # Again, load and predict with fixed random state
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)

    # assertions
    if preprocess_at_init:
        np.testing.assert_array_almost_equal(cd_load.get_config()['x_ref'], preprocess_fn(X_ref), 5)
    else:
        np.testing.assert_array_almost_equal(cd_load.get_config()['x_ref'], X_ref, 5)
    np.testing.assert_array_almost_equal(cd._detector.x_ref, cd_load._detector.x_ref, 5)
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=BinData, prefix='data_')
def test_save_fetdrift(data, tmp_path):
    """
    Test FETDrift on binary datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cd = FETDrift(X_ref,
                  p_val=P_VAL,
                  alternative='less',
                  )
    preds = cd.predict(X_h0)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)
    preds_load = cd_load.predict(X_h0)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load.x_ref)
    assert not cd_load.x_ref_preprocessed
    assert cd_load.n_features == input_dim
    assert cd_load.p_val == P_VAL
    assert cd_load.alternative == 'less'
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=CategoricalData, prefix='data_')
def test_save_chisquaredrift(data, tmp_path):
    """
    Test ChiSquareDrift on categorical datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cd = ChiSquareDrift(X_ref,
                        p_val=P_VAL,
                        )
    preds = cd.predict(X_h0)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)
    preds_load = cd_load.predict(X_h0)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load.x_ref)
    assert cd_load.n_features == input_dim
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.x_ref_categories, dict)
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)
    assert cd_load.x_ref_categories == cd.x_ref_categories


@parametrize_with_cases("data", cases=MixedData, prefix='data_')
def test_save_tabulardrift(data, tmp_path):
    """
    Test TabularDrift on mixed datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cd = TabularDrift(X_ref,
                      p_val=P_VAL,
                      categories_per_feature={0: None},
                      )
    preds = cd.predict(X_h0)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)
    preds_load = cd_load.predict(X_h0)

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load.x_ref)
    assert cd_load.n_features == input_dim
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.x_ref_categories, dict)
    assert cd_load.x_ref_categories == cd.x_ref_categories
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize('optimizer', [None, "Adam"], indirect=True)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_classifierdrift(data, optimizer, classifier_model, backend, tmp_path, seed):  # noqa: F811
    """
    Test ClassifierDrift on continuous datasets.
    """
    if backend not in ('tensorflow', 'pytorch', 'sklearn'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and predict

    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = ClassifierDrift(X_ref,
                             model=classifier_model,
                             p_val=P_VAL,
                             optimizer=optimizer,
                             n_folds=5,
                             backend=backend,
                             train_size=None)
        preds = cd.predict(X_h0)  # noqa: F841
    save_detector(cd, tmp_path)

    # Load detector and make another prediction
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)  # noqa: F841

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load._detector.x_ref)
    assert isinstance(cd_load._detector.skf, StratifiedKFold)
    assert cd_load._detector.p_val == P_VAL
    if backend != 'sklearn':
        assert isinstance(cd_load._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.model, tf.keras.Model)
    elif backend == 'pytorch':
        assert isinstance(cd_load._detector.model, nn.Module)
    elif backend == 'sklearn':
        assert isinstance(cd_load._detector.model, sklearn.base.BaseEstimator)
    # TODO - detector still not deterministic, investigate in future
    # assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    # assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_spotthediff(data, classifier_model, backend, tmp_path, seed):  # noqa: F811
    """
    Test SpotTheDiffDrift on continuous datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = SpotTheDiffDrift(X_ref,
                              p_val=P_VAL,
                              n_folds=5,
                              train_size=None,
                              backend=backend)
        preds = cd.predict(X_h0)   # noqa: F841
    save_detector(cd, tmp_path)

    # Load detector and make another prediction
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)   # noqa: F841

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load._detector._detector.x_ref)
    assert isinstance(cd_load._detector._detector.skf, StratifiedKFold)
    assert cd_load._detector._detector.p_val == P_VAL
    assert isinstance(cd_load._detector._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector._detector.model, tf.keras.Model)
    elif backend == 'pytorch':
        assert isinstance(cd_load._detector._detector.model, nn.Module)
    # TODO - detector still not deterministic, investigate in future
    # assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    # assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize('deep_kernel', [
        {'kernel_a': 'rbf', 'eps': 0.01}  # Default for kernel_a
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_learnedkernel(data, deep_kernel, backend, tmp_path, seed):  # noqa: F811
    """
    Test LearnedKernelDrift on continuous datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch', 'keops'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = LearnedKernelDrift(X_ref,
                                deep_kernel,
                                p_val=P_VAL,
                                backend=backend,
                                train_size=0.7,
                                num_workers=0)
        preds = cd.predict(X_h0)  # noqa: F841
    save_detector(cd, tmp_path)
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)  # noqa: F841

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load._detector.x_ref)
    assert not cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.kernel, DeepKernel_tf)
    elif backend == 'pytorch':
        assert isinstance(cd_load._detector.kernel, DeepKernel_pt)
    else:  # backend == keops
        assert isinstance(cd_load._detector.kernel, DeepKernel_ke)
    # TODO: Not yet deterministic
    # assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    # assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize('kernel', [
        None,  # Default kernel
        {'sigma': 0.5, 'trainable': False},  # pass kernels as GaussianRBF objects, with default sigma_median fn
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_contextmmddrift(data, kernel, backend, tmp_path, seed):  # noqa: F811
    """
    Test ContextMMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and make predictions
    X_ref, X_h0 = data
    C_ref, C_h0 = (X_ref[:, 0] + 1).reshape(-1, 1), (X_h0[:, 0] + 1).reshape(-1, 1)
    with fixed_seed(seed):
        cd = ContextMMDDrift(X_ref,
                             C_ref,
                             p_val=P_VAL,
                             backend=backend,
                             preprocess_fn=preprocess_simple,
                             n_permutations=N_PERMUTATIONS,
                             preprocess_at_init=True,
                             x_kernel=kernel,
                             c_kernel=kernel
                             )
        preds = cd.predict(X_h0, C_h0)
    save_detector(cd, tmp_path)

    # Load and make another prediction
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0, C_h0)

    # assertions
    np.testing.assert_array_equal(preprocess_simple(X_ref), cd_load._detector.x_ref)
    np.testing.assert_array_equal(C_ref, cd_load._detector.c_ref)
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.__name__ == 'preprocess_simple'
    assert cd._detector.x_kernel.sigma == cd_load._detector.x_kernel.sigma
    assert cd._detector.c_kernel.sigma == cd_load._detector.c_kernel.sigma
    assert cd._detector.x_kernel.init_sigma_fn == cd_load._detector.x_kernel.init_sigma_fn
    assert cd._detector.c_kernel.init_sigma_fn == cd_load._detector.c_kernel.init_sigma_fn
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_classifieruncertaintydrift(data, classifier_model, backend, tmp_path, seed):  # noqa: F811
    """ Test ClassifierDrift on continuous datasets."""
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = ClassifierUncertaintyDrift(X_ref,
                                        model=classifier_model,
                                        p_val=P_VAL,
                                        backend=backend,
                                        preds_type='probs',
                                        uncertainty_type='entropy')
        preds = cd.predict(X_h0)  # noqa: F841
    save_detector(cd, tmp_path)

    # Load detector and make another prediction
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)  # noqa: F841

    # Assert
    np.testing.assert_array_equal(cd._detector.preprocess_fn(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.p_val == P_VAL
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
@parametrize('regressor', [encoder_dropout_model])
def test_save_regressoruncertaintydrift(data, regressor, backend, tmp_path, seed):
    """ Test RegressorDrift on continuous datasets."""
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = RegressorUncertaintyDrift(X_ref,
                                       model=regressor,
                                       p_val=P_VAL,
                                       backend=backend,
                                       uncertainty_type='mc_dropout'
                                       )
        preds = cd.predict(X_h0)  # noqa: F841
    save_detector(cd, tmp_path)

    # Load detector and make another prediction
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)  # noqa: F841

    # Assert
    np.testing.assert_array_equal(cd._detector.preprocess_fn(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.p_val == P_VAL
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize('kernel', [
        None,  # Use default kernel
        {'sigma': 0.5, 'trainable': False},  # pass kernel as object
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_onlinemmddrift(data, kernel, preprocess_uae, backend, tmp_path, seed):  # noqa: F811
    """
    Test MMDDriftOnline on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = MMDDriftOnline(X_ref,
                            ert=ERT,
                            backend=backend,
                            preprocess_fn=preprocess_uae,
                            n_bootstraps=N_BOOTSTRAPS,
                            kernel=kernel,
                            window_size=WINDOW_SIZE
                            )
        stats = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:  # test stats garbage until window full
                stats.append(pred['data']['test_stat'])
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        stats_load = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:
                stats_load.append(pred['data']['test_stat'])

    # assertions
    np.testing.assert_array_equal(preprocess_uae(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.n_bootstraps == N_BOOTSTRAPS
    assert cd_load._detector.ert == ERT
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert cd._detector.kernel.sigma == cd_load._detector.kernel.sigma
    assert cd._detector.kernel.init_sigma_fn == cd_load._detector.kernel.init_sigma_fn
    np.testing.assert_array_equal(stats, stats_load)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_onlinelsdddrift(data, preprocess_uae, backend, tmp_path, seed):
    """
    Test LSDDDriftOnline on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = LSDDDriftOnline(X_ref,
                             ert=ERT,
                             backend=backend,
                             preprocess_fn=preprocess_uae,
                             n_bootstraps=N_BOOTSTRAPS,
                             window_size=WINDOW_SIZE
                             )
        stats = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:  # test stats garbage until window full
                stats.append(pred['data']['test_stat'])
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        stats_load = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:
                stats_load.append(pred['data']['test_stat'])

    # assertions
    np.testing.assert_array_almost_equal(preprocess_uae(X_ref), cd_load.get_config()['x_ref'], 5)
    assert cd_load._detector.n_bootstraps == N_BOOTSTRAPS
    assert cd_load._detector.ert == ERT
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert cd._detector.kernel.sigma == cd_load._detector.kernel.sigma
    assert cd._detector.kernel.init_sigma_fn == cd_load._detector.kernel.init_sigma_fn
    np.testing.assert_array_equal(stats, stats_load)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_onlinecvmdrift(data, preprocess_uae, tmp_path, seed):
    """
    Test CVMDriftOnline on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = CVMDriftOnline(X_ref,
                            ert=ERT,
                            preprocess_fn=preprocess_uae,
                            n_bootstraps=N_BOOTSTRAPS,
                            window_sizes=[WINDOW_SIZE]
                            )
        stats = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:  # test stats garbage until at least one window full
                stats.append(pred['data']['test_stat'])
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        stats_load = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:  # test stats garbage until at least one window full
                stats_load.append(pred['data']['test_stat'])

    # assertions
    np.testing.assert_array_almost_equal(preprocess_uae(X_ref), cd_load.get_config()['x_ref'], 5)
    assert cd_load.n_bootstraps == N_BOOTSTRAPS
    assert cd_load.ert == ERT
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(stats, stats_load)


@parametrize_with_cases("data", cases=BinData, prefix='data_')
def test_save_onlinefetdrift(data, tmp_path, seed):
    """
    Test FETDriftOnline on binary datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = FETDriftOnline(X_ref,
                            ert=ERT,
                            n_bootstraps=N_BOOTSTRAPS,
                            window_sizes=[WINDOW_SIZE]
                            )
        stats = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:  # test stats garbage until at least one window full
                stats.append(pred['data']['test_stat'])
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        stats_load = []
        for i, x_t in enumerate(X_h0):
            pred = cd.predict(x_t)
            if i >= WINDOW_SIZE:  # test stats garbage until at least one window full
                stats_load.append(pred['data']['test_stat'])

    # assertions
    np.testing.assert_array_equal(X_ref, cd_load.get_config()['x_ref'])
    assert cd_load.n_bootstraps == N_BOOTSTRAPS
    assert cd_load.ert == ERT
    np.testing.assert_array_almost_equal(stats, stats_load, 4)


@parametrize("detector", [MMDDriftOnline, LSDDDriftOnline])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_multivariate_online_state(detector, data, backend, seed, tmp_path):
    """
    Test the saving (and loading) of multivariate online detectors' state via `save_detector`.
    """
    # Skip if backend not `tensorflow` or `pytorch`
    if backend not in ('tensorflow', 'pytorch'):
        pytest.skip("Detector doesn't have this backend")

    # Init detector and make prediction to update state
    X_ref, X_h0 = data
    with fixed_seed(seed):
        dd = detector(X_ref, ert=100, window_size=10, backend=backend)

    # Run for 10 time-steps
    test_stats = []
    for t, x_t in enumerate(X_h0[:10]):
        if t == 5:
            # Save detector (with state)
            save_detector(dd, tmp_path)
        test_stats.append(dd.predict(x_t)['data']['test_stat'])

    # Check state file created
    assert dd._detector.state_dir == tmp_path.joinpath('state')

    # Load
    with fixed_seed(seed):
        dd_new = load_detector(tmp_path)
    # Check attributes and compare predictions at t=5
    assert dd_new.t == 5
    if detector == LSDDDriftOnline:  # Often a small (~1e-6) difference in LSDD test stats post-load # TODO - why?
        np.testing.assert_array_almost_equal(dd_new.predict(X_h0[5])['data']['test_stat'], test_stats[5], 5)
    else:
        np.testing.assert_array_equal(dd_new.predict(X_h0[5])['data']['test_stat'], test_stats[5])

    # Check that error raised if no state file inside `state/` dir
    for child in tmp_path.joinpath('state').glob('*'):
        if child.is_file():
            child.unlink()
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path)


@parametrize("detector", [CVMDriftOnline])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_cvm_online_state(detector, data, tmp_path):
    """
    Test the saving (and loading) of the CVM online detector's state via `save_detector`.
    """
    # Init detector and make prediction to update state
    X_ref, X_h0 = data
    dd = detector(X_ref, ert=100, window_sizes=[10])

    # Run for 10 time-steps
    test_stats = []
    for t, x_t in enumerate(X_h0[:10]):
        if t == 5:
            # Save detector (with state)
            save_detector(dd, tmp_path)
        test_stats.append(dd.predict(x_t)['data']['test_stat'])

    # Check state file created
    assert dd.state_dir == tmp_path.joinpath('state')

    # Load
    dd_new = load_detector(tmp_path)
    # Check attributes and compare predictions at t=5
    assert dd_new.t == 5
    np.testing.assert_array_equal(dd_new.predict(X_h0[5])['data']['test_stat'], test_stats[5])

    # Check that error raised if no state file inside `state/` dir
    for child in tmp_path.joinpath('state').glob('*'):
        if child.is_file():
            child.unlink()
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path)


@parametrize("detector", [FETDriftOnline])
@parametrize_with_cases("data", cases=BinData, prefix='data_')
def test_save_fet_online_state(detector, data, tmp_path):
    """
    Test the saving (and loading) of the FET online detector's state via `save_detector`.
    """
    # Init detector and make prediction to update state
    X_ref, X_h0 = data
    dd = detector(X_ref, ert=100, window_sizes=[10])

    # Run for 10 time-steps
    test_stats = []
    for t, x_t in enumerate(X_h0[:10]):
        if t == 5:
            # Save detector (with state)
            save_detector(dd, tmp_path)
        test_stats.append(dd.predict(x_t)['data']['test_stat'])

    # Check state file created
    assert dd.state_dir == tmp_path.joinpath('state')

    # Load
    dd_new = load_detector(tmp_path)
    # Check attributes and compare predictions at t=5
    assert dd_new.t == 5
    np.testing.assert_array_equal(dd_new.predict(X_h0[5])['data']['test_stat'], test_stats[5])

    # Check that error raised if no state file inside `state/` dir
    for child in tmp_path.joinpath('state').glob('*'):
        if child.is_file():
            child.unlink()
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_online_state_t0(data, tmp_path):
    """
    Test that state is not saved when t=0.
    """
    # Init detector
    X_ref, X_h0 = data
    dd = CVMDriftOnline(X_ref, ert=100, window_sizes=[10])
    # Check state NOT saved when t=0
    state_dir = tmp_path.joinpath('state')
    save_detector(dd, tmp_path)
    assert not state_dir.is_dir()
    # Check state IS saved when t>0
    dd.predict(X_h0[0])
    save_detector(dd, tmp_path)
    assert state_dir.is_dir()


@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd)
def test_load_absolute(data, tmp_path):
    """
    Test that load_detector() works with absolute paths in config.
    """
    # Init detector and save
    X_ref, X_h0 = data
    cd = KSDrift(X_ref, p_val=P_VAL)
    save_detector(cd, tmp_path)
    # Write a new cfg file elsewhere, with x_ref reference inside it an absolute path to original x_ref location
    cfg = read_config(tmp_path.joinpath('config.toml'))
    x_ref_path = tmp_path.joinpath(Path(cfg['x_ref'])).resolve()  # Absolute path for x_ref
    cfg['x_ref'] = x_ref_path.as_posix()  # we always write paths to config.toml as Posix not Windows paths
    new_cfg_dir = tmp_path.joinpath('new_config_dir')
    new_cfg_dir.mkdir()
    write_config(cfg, new_cfg_dir)

    # Reload
    cd_new = load_detector(new_cfg_dir)

    # Assertions
    np.testing.assert_array_equal(cd.x_ref, cd_new.x_ref)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_version_warning(data, tmp_path):
    """
    Test that a version mismatch warning is raised if a detector is loaded from a config generated with a
    different alibi_detect version, then saved, then loaded again (warning is still expected on final load).

    This is only tested on one detector since the functionality lies outside of the actual detector classes.
    """
    X_ref, X_h0 = data
    cd = KSDrift(X_ref, p_val=P_VAL)
    # First save (just to create a config)
    save_detector(cd, tmp_path)
    # Emulate version mismatch
    cfg = read_config(tmp_path.joinpath('config.toml'))
    cfg['meta']['version'] = '0.1.x'
    _ = write_config(cfg, tmp_path)
    # Reload and save again
    cd = load_detector(tmp_path)
    save_detector(cd, tmp_path)
    # Check saved config contains a "version_warning"
    cfg = read_config(tmp_path.joinpath('config.toml'))
    assert cfg['meta']['version_warning']
    # Final load (we expect a warning to be raised here)
    with pytest.warns(Warning):  # error will be raised if a warning IS NOT raised
        cd_new = load_detector(tmp_path)
        assert cd_new.meta.get('version_warning', False)


@parametrize('kernel', [
        {'sigma': 0.5, 'trainable': False, 'init_sigma_fn': None},
        {'sigma': [0.5, 0.8], 'trainable': False, 'init_sigma_fn': None},
        {'sigma': None, 'trainable': True, 'init_sigma_fn': None},
    ], indirect=True
)
def test_save_kernel(kernel, backend, tmp_path):  # noqa: F811
    """
    Unit test for _save/_load_kernel_config, when kernel is a GaussianRBF kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Save kernel to config
    filepath = tmp_path
    filename = Path('mykernel')
    cfg_kernel = _save_kernel_config(kernel, filepath, filename)
    cfg_kernel = KernelConfig(**cfg_kernel).dict()  # Pass through validator to test, and coerce sigma to Tensor
    if kernel.__class__.__name__ == 'GaussianRBF':
        assert cfg_kernel['src'] == '@utils.' + backend + '.kernels.GaussianRBF'
    else:
        assert Path(cfg_kernel['src']).suffix == '.dill'
    assert cfg_kernel['trainable'] == kernel.trainable
    if not kernel.trainable and cfg_kernel['sigma'] is not None:
        np.testing.assert_array_almost_equal(cfg_kernel['sigma'], kernel.sigma, 6)

    # Resolve and load config (_load_kernel_config is called within resolve_config)
    cfg = {'kernel': cfg_kernel, 'backend': backend}
    _prepend_cfg_filepaths(cfg, tmp_path)
    kernel_loaded = resolve_config(cfg, tmp_path)['kernel']

    # Call kernels
    if backend == 'tensorflow':
        X = tf.random.normal((10, 1), dtype=tf.float32)
    elif backend == 'pytorch':
        X = torch.randn((10, 1), dtype=torch.float32)
    else:  # backend == 'keops'
        X = torch.randn((10, 1), dtype=torch.float32)
        X = LazyTensor(X[None, :])
    kernel(X, X)
    kernel_loaded(X, X)

    # Final checks
    assert type(kernel_loaded) == type(kernel)  # noqa: E721
    if backend == 'tensorflow':
        np.testing.assert_array_almost_equal(np.array(kernel_loaded.sigma), np.array(kernel.sigma), 5)
    else:
        np.testing.assert_array_almost_equal(kernel_loaded.sigma.detach().numpy(), kernel.sigma.detach().numpy(), 5)
    assert kernel_loaded.trainable == kernel.trainable
    assert kernel_loaded.init_sigma_fn == kernel.init_sigma_fn


# `data` passed below as needed in encoder_model, which is used in deep_kernel
@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd)
@parametrize('deep_kernel', [
        {'kernel_a': 'rbf', 'kernel_b': 'rbf', 'eps': 'trainable'},  # Default for kernel_a and kernel_b, trainable eps
        {'kernel_a': {'trainable': True}, 'kernel_b': 'rbf', 'eps': 0.01},  # Explicit kernel_a, fixed eps
    ], indirect=True
)
def test_save_deepkernel(data, deep_kernel, backend, tmp_path):  # noqa: F811
    """
    Unit test for _save/_load_kernel_config, when kernel is a DeepKernel kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Get data dim
    if backend == 'tensorflow':
        X = tf.random.normal((10, 1), dtype=tf.float32)
    elif backend == 'pytorch':
        X = torch.randn((10, 1), dtype=torch.float32)
    else:  # backend == 'keops'
        X = torch.randn((10, 1), dtype=torch.float32)
        X = LazyTensor(X[None, :])
#    X, _ = data
    input_shape = (X.shape[1],)

    # Save kernel to config
    filepath = tmp_path
    filename = 'mykernel'
    cfg_kernel = _save_kernel_config(deep_kernel, filepath, filename)
    cfg_kernel['proj'], _ = _save_model_config(cfg_kernel['proj'], base_path=filepath, input_shape=input_shape)
    cfg_kernel = _path2str(cfg_kernel)
    cfg_kernel['proj'] = ModelConfig(**cfg_kernel['proj']).dict()  # Pass thru ModelConfig to set `layers` etc
    cfg_kernel = DeepKernelConfig(**cfg_kernel).dict()  # pydantic validation
    assert cfg_kernel['proj']['src'] == 'model'
    assert cfg_kernel['proj']['custom_objects'] is None
    assert cfg_kernel['proj']['layer'] is None

    # Resolve and load config
    cfg = {'kernel': cfg_kernel, 'backend': backend}
    kernel_loaded = resolve_config(cfg, tmp_path)['kernel']  # implicitly calls _load_kernel_config

    # Call kernels
    deep_kernel.kernel_a(X, X)
    deep_kernel.kernel_b(X, X)
    kernel_loaded.kernel_a(X, X)
    kernel_loaded.kernel_b(X, X)

    # Final checks
    assert isinstance(kernel_loaded.proj, (torch.nn.Module, tf.keras.Model))
    if backend == 'tensorflow':
        assert pytest.approx(deep_kernel.eps.numpy(), abs=1e-4) == kernel_loaded.eps.numpy()
    else:
        assert pytest.approx(deep_kernel.eps.detach().numpy(), abs=1e-4) == kernel_loaded.eps.detach().numpy()
    assert kernel_loaded.kernel_a.sigma == deep_kernel.kernel_a.sigma
    assert kernel_loaded.kernel_b.sigma == deep_kernel.kernel_b.sigma


@parametrize('preprocess_fn', [preprocess_uae, preprocess_hiddenoutput])
@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
def test_save_preprocess_drift(data, preprocess_fn, tmp_path, backend):
    """
    Test saving/loading of the inbuilt `preprocess_drift` preprocessing functions when containing a `model`, with the
    `model` either being a simple tf/torch model, or a `HiddenOutput` class.
    """
    registry_str = 'tensorflow' if backend == 'tensorflow' else 'pytorch'
    # Save preprocess_fn to config
    filepath = tmp_path
    X_ref, X_h0 = data
    input_shape = (X_ref.shape[1],)
    cfg_preprocess = _save_preprocess_config(preprocess_fn, input_shape=input_shape, filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()  # pydantic validation
    assert cfg_preprocess['src'] == '@cd.' + registry_str + '.preprocess.preprocess_drift'
    assert cfg_preprocess['model']['src'] == 'preprocess_fn/model'
    # TODO - check layer details here once implemented
    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess, 'backend': backend}
    preprocess_fn_load = resolve_config(cfg, tmp_path)['preprocess_fn']  # tests _load_preprocess_config implicitly
    if backend == 'tensorflow':
        assert preprocess_fn_load.func.__name__ == 'preprocess_drift'
        assert isinstance(preprocess_fn_load.keywords['model'], tf.keras.Model)
    else:  # pytorch and keops backend
        assert preprocess_fn_load.func.__name__ == 'preprocess_drift'
        assert isinstance(preprocess_fn_load.keywords['model'], nn.Module)


@parametrize('preprocess_fn', [preprocess_simple, preprocess_simple_with_kwargs])
def test_save_preprocess_custom(preprocess_fn, tmp_path):
    """
    Test saving/loading of custom preprocessing functions, without and with kwargs.
    """
    # Save preprocess_fn to config
    filepath = tmp_path
    cfg_preprocess = _save_preprocess_config(preprocess_fn, input_shape=None, filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()  # pydantic validation

    assert tmp_path.joinpath(cfg_preprocess['src']).is_file()
    assert cfg_preprocess['src'] == os.path.join('preprocess_fn', 'function.dill')
    if isinstance(preprocess_fn, partial):  # kwargs expected
        assert cfg_preprocess['kwargs'] == preprocess_fn.keywords
    else:  # no kwargs expected
        assert cfg_preprocess['kwargs'] == {}

    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess}
    preprocess_fn_load = resolve_config(cfg, tmp_path)['preprocess_fn']  # tests _load_preprocess_config implicitly
    if isinstance(preprocess_fn, partial):
        assert preprocess_fn_load.func == preprocess_fn.func
        assert preprocess_fn_load.keywords == preprocess_fn.keywords
    else:
        assert preprocess_fn_load == preprocess_fn


@parametrize('preprocess_fn', [preprocess_nlp])
@parametrize_with_cases("data", cases=TextData.movie_sentiment_data, prefix='data_')
def test_save_preprocess_nlp(data, preprocess_fn, tmp_path, backend):
    """
    Test saving/loading of the inbuilt `preprocess_drift` preprocessing functions when containing a `model`, text
    `tokenizer` and text `embedding` model.
    """
    registry_str = 'tensorflow' if backend == 'tensorflow' else 'pytorch'
    # Save preprocess_fn to config
    filepath = tmp_path
    cfg_preprocess = _save_preprocess_config(preprocess_fn,
                                             input_shape=(768,),  # hardcoded to bert-base-cased for now
                                             filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()  # pydantic validation
    assert cfg_preprocess['src'] == '@cd.' + registry_str + '.preprocess.preprocess_drift'
    assert cfg_preprocess['embedding']['src'] == 'preprocess_fn/embedding'
    assert cfg_preprocess['tokenizer']['src'] == 'preprocess_fn/tokenizer'
    assert tmp_path.joinpath(cfg_preprocess['preprocess_batch_fn']).is_file()
    assert cfg_preprocess['preprocess_batch_fn'] == os.path.join('preprocess_fn', 'preprocess_batch_fn.dill')

    if isinstance(preprocess_fn.keywords['model'], (TransformerEmbedding_tf, TransformerEmbedding_pt)):
        assert cfg_preprocess['model'] is None
    else:
        assert cfg_preprocess['model']['src'] == 'preprocess_fn/model'

    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess, 'backend': backend}
    preprocess_fn_load = resolve_config(cfg, tmp_path)['preprocess_fn']  # tests _load_preprocess_config implicitly
    assert isinstance(preprocess_fn_load.keywords['tokenizer'], type(preprocess_fn.keywords['tokenizer']))
    assert isinstance(preprocess_fn_load.keywords['model'], type(preprocess_fn.keywords['model']))
    if isinstance(preprocess_fn.keywords['model'], (TransformerEmbedding_tf, TransformerEmbedding_pt)):
        emb = preprocess_fn.keywords['model']
        emb_load = preprocess_fn_load.keywords['model']
    else:
        if backend == 'tensorflow':
            emb = preprocess_fn.keywords['model'].encoder.layers[0]
            emb_load = preprocess_fn_load.keywords['model'].encoder.layers[0]
        else:  # pytorch and keops backends
            emb = list(preprocess_fn.keywords['model'].encoder.children())[0]
            emb_load = list(preprocess_fn_load.keywords['model'].encoder.children())[0]
    assert isinstance(emb_load.model, type(emb.model))
    assert emb_load.emb_type == emb.emb_type
    assert emb_load.hs_emb.keywords['layers'] == emb.hs_emb.keywords['layers']


def test_nested_value():
    """
    Unit test for _get_nested_value and _set_nested_value.
    """
    dict1 = {'dict2': {'dict3': {}}}
    _set_nested_value(dict1, ['dict2', 'dict3', 'a string'], 'hello')
    _set_nested_value(dict1, ['a float'], 42.0)
    _set_nested_value(dict1, ['dict2', 'a list'], [1, 2, 3])
    assert _get_nested_value(dict1, ['dict2', 'dict3', 'a string']) == dict1['dict2']['dict3']['a string']
    assert _get_nested_value(dict1, ['a float']) == dict1['a float']
    assert _get_nested_value(dict1, ['dict2', 'a list']) == dict1['dict2']['a list']


def test_replace():
    """
    A unit test for _replace.
    """
    dict1 = {
        'key1': 'key1',
        'key7': None,
        'dict2': {
            'key2': 'key2',
            'key4': None,
            'dict3': {
                'key5': 'key5',
                'key6': None
            }
        }
    }
    new_dict = _replace(dict1, None, 'None')
    assert new_dict['key7'] == 'None'
    assert new_dict['dict2']['key4'] == 'None'
    assert new_dict['dict2']['dict3']['key6'] == 'None'
    assert new_dict['key1'] == dict1['key1']


def test_path2str(tmp_path):
    """
    A unit test for _path2str.
    """
    cfg = {
        'dict': {'a path': tmp_path}
    }
    cfg_rel = _path2str(cfg)
    rel_path = cfg_rel['dict']['a path']
    assert isinstance(rel_path, str)
    assert rel_path == str(tmp_path.as_posix())

    cfg_abs = _path2str(cfg, absolute=True)
    abs_path = cfg_abs['dict']['a path']
    assert isinstance(abs_path, str)
    assert abs_path == str(tmp_path.resolve().as_posix())


def test_int2str_keys():
    """
    A unit test for _int2str_keys
    """
    cfg = {
        'dict': {'0': 'A', '1': 3, 2: 'C'},
        3: 'D',
        '4': 'E'
    }
    cfg_fixed = _int2str_keys(cfg)

    # Check all str keys changed to int
    assert cfg['dict'].pop(2) == cfg_fixed['dict'].pop('2')
    assert cfg.pop(3) == cfg_fixed.pop('3')

    # Check remaining items untouched
    assert cfg == cfg_fixed

    assert cfg


def generic_function(x: float, add: float = 0.0, invert: bool = True):
    if invert:
        return 1/x + add
    else:
        return x + add


@parametrize('function', [generic_function])
def test_serialize_function_partial(function, tmp_path):
    """
    Unit tests for _serialize_function, with a functools.partial function.
    """
    partial_func = partial(function, invert=False, add=1.0)
    src, kwargs = _serialize_object(partial_func, base_path=tmp_path, local_path=Path('function'))
    filepath = tmp_path.joinpath('function.dill')
    assert filepath.is_file()
    with open(filepath, 'rb') as f:
        partial_func_load = dill.load(f)
    x = 2.0
    assert partial_func_load(x, **kwargs) == partial_func(x)


def test_serialize_function_registry(tmp_path):
    """
    Unit tests for _serialize_function, with a registered function.
    """
    registry_ref = 'cd.tensorflow.preprocess.preprocess_drift'
    function = registry.get(registry_ref)
    src, kwargs = _serialize_object(function, base_path=tmp_path, local_path=Path('function'))
    assert kwargs == {}
    assert src == '@' + registry_ref


def test_registry_get():
    """
    Unit test for alibi_detect.utils.registry.get(). This will make more sense once we have a more automated
    process for pre-registering alibi-detect objects, as then can compare against list of objects we wish to register.
    """
    for k, v in REGISTERED_OBJECTS.items():
        obj = registry.get(k)
        assert type(obj) == type(v)  # noqa: E721


def test_set_dtypes(backend):
    """
    Unit test to test _set_dtypes.
    """
    if backend == 'tensorflow':
        dtype = 'tf.float32'
    elif backend == 'pytorch':
        dtype = 'torch.float32'
    else:
        pytest.skip('Only test set_dtypes for tensorflow and pytorch.')

    cfg = {
        'preprocess_fn': {
            'dtype': dtype
        }
    }
    _set_dtypes(cfg)
    dtype_resolved = cfg['preprocess_fn']['dtype']
    if backend == 'tensorflow':
        assert dtype_resolved == tf.float32
    elif backend == 'pytorch':
        assert dtype_resolved == torch.float32


def test_cleanup(tmp_path):
    """
    Test that the filepath given to save_detector is deleted in the event of an error whilst saving.
    Also check that the error is caught and raised.
    """
    # Detector save/load
    X_ref = np.random.normal(size=(5, 1))
    cd = KSDrift(X_ref)

    # Add a garbage preprocess_fn to cause an error
    cd.preprocess_fn = cd.x_ref

    # Save, catch and check error
    with pytest.raises(RuntimeError):
        save_detector(cd, tmp_path)

    # Check `filepath` is deleted
    assert not tmp_path.is_dir()


@pytest.mark.parametrize('backend, device', [
    ('pytorch', 'cpu'),
    ('pytorch', 'gpu'),
    ('pytorch', 'cuda'),
    ('pytorch', 'cuda:0'),
    ('pytorch', torch.device('cuda')),
    ('pytorch', torch.device('cuda:0')),
    ('tensorflow', None),
])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_detector_device(backend, device, data, tmp_path, classifier_model):  # noqa: F811
    """
    Test saving a Detector with different pytorch device options.

    Save using `save_detector` and load using `load_detector`, with assertions checking that the reinstantiated
    detector is equivalent. Also check that the detector config toml device string is correct.
    """
    X_ref, X_h0 = data
    detector = ClassifierDrift(
        X_ref,
        backend=backend,
        model=classifier_model,
        device=device
    )
    save_detector(detector, tmp_path)
    detector_config = toml.load(tmp_path / 'config.toml')
    loaded_detector = load_detector(tmp_path)
    if backend == 'tensorflow':
        assert detector_config['device'] == 'None'
    else:
        assert detector_config['device'] in {'cpu', 'gpu', 'cuda'}
        assert loaded_detector._detector.device in {torch.device('cpu'), torch.device('cuda')}
