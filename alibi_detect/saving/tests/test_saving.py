# type: ignore
"""
Tests for saving/loading of detectors via config.toml files.

Internal functions such as save_kernel/load_kernel_config etc are also tested.
"""
# TODO future - test pytorch save/load functionality
# TODO (could/should also add tests to backend-specific submodules)
from functools import partial
from pathlib import Path
from typing import Callable

import toml
import dill
import numpy as np
import pytest
import scipy
import tensorflow as tf
import torch

from .datasets import BinData, CategoricalData, ContinuousData, MixedData, TextData
from alibi_detect.utils._random import fixed_seed
from packaging import version
from pytest_cases import fixture, param_fixture, parametrize, parametrize_with_cases
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

from alibi_detect.cd import (ChiSquareDrift, ClassifierUncertaintyDrift, RegressorUncertaintyDrift,
                             ClassifierDrift, FETDrift, KSDrift, LearnedKernelDrift, LSDDDrift, MMDDrift,
                             SpotTheDiffDrift, TabularDrift, ContextMMDDrift, MMDDriftOnline, LSDDDriftOnline,
                             CVMDriftOnline, FETDriftOnline)
from alibi_detect.cd.pytorch import HiddenOutput as HiddenOutput_pt
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_pt
from alibi_detect.cd.tensorflow import UAE as UAE_tf
from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.models.pytorch import TransformerEmbedding as TransformerEmbedding_pt
from alibi_detect.models.tensorflow import TransformerEmbedding as TransformerEmbedding_tf
from alibi_detect.saving import (load_detector, read_config, registry,
                                 resolve_config, save_detector, write_config)
from alibi_detect.saving.loading import (_get_nested_value, _load_model_config, _load_optimizer_config, _replace,
                                         _set_dtypes, _set_nested_value, _prepend_cfg_filepaths)
from alibi_detect.saving.saving import _serialize_object
from alibi_detect.saving.saving import (_path2str, _int2str_keys, _save_kernel_config, _save_model_config,
                                        _save_preprocess_config)
from alibi_detect.saving.schemas import DeepKernelConfig, KernelConfig, ModelConfig, PreprocessConfig
from alibi_detect.utils.pytorch.kernels import DeepKernel as DeepKernel_pt
from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_pt
from alibi_detect.utils.tensorflow.kernels import DeepKernel as DeepKernel_tf
from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf

if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift

backend = param_fixture("backend", ['tensorflow'])
P_VAL = 0.05
ERT = 10
N_PERMUTATIONS = 10
N_BOOTSTRAPS = 100
WINDOW_SIZE = 5
LATENT_DIM = 2  # Must be less than input_dim set in ./datasets.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


@fixture
def encoder_model(backend, current_cases):
    """
    An untrained encoder of given input dimension and backend (this is a "custom" model, NOT an Alibi Detect UAE).
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']

    if backend == 'tensorflow':
        model = tf.keras.Sequential(
               [
                   tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                   tf.keras.layers.Dense(5, activation=tf.nn.relu),
                   tf.keras.layers.Dense(LATENT_DIM, activation=None)
               ]
           )
    else:
        raise NotImplementedError('`pytorch` tests not implemented.')
    return model


@fixture
def encoder_dropout_model(backend, current_cases):
    """
    An untrained encoder with dropout, of given input dimension and backend.

    TODO: consolidate this model (and encoder_model above) with models like that in test_model_uncertainty.py
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']

    if backend == 'tensorflow':
        model = tf.keras.Sequential(
               [
                   tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                   tf.keras.layers.Dense(5, activation=tf.nn.relu),
                   tf.keras.layers.Dropout(0.5),
                   tf.keras.layers.Dense(LATENT_DIM, activation=None)
               ]
           )
    else:
        raise NotImplementedError('`pytorch` tests not implemented.')
    return model


@fixture
def preprocess_custom(encoder_model, backend):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if backend == 'tensorflow':
        preprocess_fn = partial(preprocess_drift_tf, model=encoder_model)
    else:
        preprocess_fn = partial(preprocess_drift_pt, model=encoder_model)
    return preprocess_fn


@fixture
def kernel(request, backend):
    """
    Gaussian RBF kernel for given backend. Settings are parametrised in the test function.
    """
    kernel = request.param
    if kernel is None:
        pass
    elif isinstance(kernel, dict):  # dict of kwargs
        if backend == 'tensorflow':
            kernel = GaussianRBF_tf(**kernel)
        elif backend == 'pytorch':
            kernel = GaussianRBF_pt(**kernel)
    return kernel


@fixture
def deep_kernel(request, backend, encoder_model):
    """
    Deep kernel, built using the `encoder_model` fixture for the projection, and using the kernel_a and eps
    parametrised in the test function.
    """
    # Get DeepKernel options
    kernel_a = request.param.get('kernel_a', 'rbf')
    kernel_b = request.param.get('kernel_b', 'rbf')
    eps = request.param.get('eps', 'trainable')

    # Proj model (backend managed in encoder_model fixture)
    proj = encoder_model

    # Build DeepKernel
    if backend == 'tensorflow':
        kernel_a = GaussianRBF_tf(**kernel_a) if isinstance(kernel_a, dict) else kernel_a
        kernel_a = GaussianRBF_tf(**kernel_b) if isinstance(kernel_b, dict) else kernel_b
        deep_kernel = DeepKernel_tf(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
    elif backend == 'pytorch':
        raise NotImplementedError('`pytorch` tests not implemented.')
    else:
        raise ValueError('`backend` not valid.')
    return deep_kernel


@fixture
def classifier(backend, current_cases):
    """
    Classification model with given input dimension and backend.
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
        raise ValueError('`backend` not valid.')
    return model


@fixture(unpack_into=('tokenizer, embedding, max_len, enc_dim'))
@parametrize('model_name, max_len', [('bert-base-cased', 100)])
@parametrize('uae', [True, False])
def nlp_embedding_and_tokenizer(model_name, max_len, uae, backend):
    """
    A fixture to build nlp embedding and tokenizer models based on the HuggingFace pre-trained models.
    """
    backend = 'tf' if backend == 'tensorflow' else 'pt'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    X = 'A dummy string'  # this will be padded to max_len
    tokens = tokenizer(list(X[:5]), pad_to_max_length=True,
                       max_length=max_len, return_tensors=backend)

    # Load embedding model
    emb_type = 'hidden_state'
    n_layers = 8
    layers = [-_ for _ in range(1, n_layers + 1)]
    enc_dim = 32

    if backend == 'tf':
        embedding = TransformerEmbedding_tf(model_name, emb_type, layers)
        if uae:
            x_emb = embedding(tokens)
            shape = (x_emb.shape[1],)
            embedding = UAE_tf(input_layer=embedding, shape=shape, enc_dim=enc_dim)
    else:
        embedding = TransformerEmbedding_pt(model_name, emb_type, layers)
        if uae:
            x_emb = embedding(tokens)
            emb_dim = x_emb.shape[1]
            device = torch.device(DEVICE)
            embedding = torch.nn.Sequential(
                embedding,
                torch.nn.Linear(emb_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, enc_dim)
            ).to(device).eval()

    return tokenizer, embedding, max_len, enc_dim


def preprocess_simple(x: np.ndarray):
    """
    Simple function to test serialization of generic Python function within preprocess_fn.
    """
    return x*2.0


@fixture
def preprocess_nlp(embedding, tokenizer, max_len, backend):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if backend == 'tensorflow':
        preprocess_fn = partial(preprocess_drift_tf, model=embedding, tokenizer=tokenizer,
                                max_len=max_len, preprocess_batch_fn=preprocess_simple)
    else:
        preprocess_fn = partial(preprocess_drift_pt, model=embedding, tokenizer=tokenizer, max_len=max_len,
                                preprocess_batch_fn=preprocess_simple)
    return preprocess_fn


@fixture
def preprocess_hiddenoutput(classifier, backend):
    """
    Preprocess function to extract the softmax layer of a classifier (with the HiddenOutput utility function).
    """
    if backend == 'tensorflow':
        model = HiddenOutput_tf(classifier, layer=-1)
        preprocess_fn = partial(preprocess_drift_tf, model=model)
    else:
        model = HiddenOutput_pt(classifier, layer=-1)
        preprocess_fn = partial(preprocess_drift_pt, model=model)
    return preprocess_fn


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


@parametrize('preprocess_fn', [preprocess_custom, preprocess_hiddenoutput])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_ksdrift(data, preprocess_fn, tmp_path):
    """
    Test KSDrift on continuous datasets, with UAE and classifier softmax output as preprocess_fn's. Only this
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


@parametrize('preprocess_fn', [preprocess_nlp])
@parametrize_with_cases("data", cases=TextData.movie_sentiment_data, prefix='data_')
def test_save_ksdrift_nlp(data, preprocess_fn, max_len, enc_dim, tmp_path):
    """
    Test KSDrift on continuous datasets, with UAE and classifier softmax output as preprocess_fn's. Only this
    detector is tested with embedding and embedding+uae, as other detectors should see the same preprocessed data.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data['X_train'][:5], data['X_test'][:5]
    cd = KSDrift(X_ref,
                 p_val=P_VAL,
                 preprocess_fn=preprocess_fn,
                 preprocess_at_init=True,
                 input_shape=(max_len,),
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
def test_save_cvmdrift(data, preprocess_custom, tmp_path):
    """
    Test CVMDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    cd = CVMDrift(X_ref,
                  p_val=P_VAL,
                  preprocess_fn=preprocess_custom,
                  preprocess_at_init=True,
                  )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load.x_ref)
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
def test_save_mmddrift(data, kernel, preprocess_custom, backend, tmp_path, seed):
    """
    Test MMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Init detector and make predictions
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = MMDDrift(X_ref,
                      p_val=P_VAL,
                      backend=backend,
                      preprocess_fn=preprocess_custom,
                      n_permutations=N_PERMUTATIONS,
                      preprocess_at_init=True,
                      kernel=kernel,
                      configure_kernel_from_x_ref=False,
                      sigma=np.array([0.5])
                      )
        preds = cd.predict(X_h0)
    save_detector(cd, tmp_path)

    # Load and make predictions
    with fixed_seed(seed):
        cd_load = load_detector(tmp_path)
        preds_load = cd_load.predict(X_h0)

    # assertions
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector.x_ref)
    assert not cd_load._detector.infer_sigma
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert cd._detector.kernel.sigma == cd_load._detector.kernel.sigma
    assert cd._detector.kernel.init_sigma_fn == cd_load._detector.kernel.init_sigma_fn
    assert preds['data']['p_val'] == preds_load['data']['p_val']


# @parametrize('preprocess_fn', [preprocess_custom, preprocess_hiddenoutput])
@parametrize('preprocess_at_init', [True, False])
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_lsdddrift(data, preprocess_at_init, backend, tmp_path, seed):
    """
    Test LSDDDrift on continuous datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    preprocess_fn = preprocess_simple
    # TODO - TensorFlow based preprocessors currently cause in-deterministic behaviour with LSDD permutations. Replace
    # preprocess_simple with parametrized preprocess_fn's once above issue resolved.

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


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_classifierdrift(data, classifier, backend, tmp_path, seed):
    """ Test ClassifierDrift on continuous datasets."""
    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = ClassifierDrift(X_ref,
                             model=classifier,
                             p_val=P_VAL,
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
    assert isinstance(cd_load._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.model, tf.keras.Model)
    else:
        pass  # TODO
    # TODO - detector still not deterministic, investigate in future
    # assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    # assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_spotthediff(data, classifier, backend, tmp_path, seed):
    """
    Test SpotTheDiffDrift on continuous datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
    else:
        pass  # TODO
    # TODO - detector still not deterministic, investigate in future
    # assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    # assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize('deep_kernel', [
        {'kernel_a': 'rbf', 'eps': 0.01}  # Default for kernel_a
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_learnedkernel(data, deep_kernel, backend, tmp_path, seed):
    """
    Test LearnedKernelDrift on continuous datasets.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = LearnedKernelDrift(X_ref,
                                deep_kernel,
                                p_val=P_VAL,
                                backend=backend,
                                train_size=0.7)
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
    else:
        assert isinstance(cd_load._detector.kernel, DeepKernel_pt)
    # TODO: Not yet deterministic
    # assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    # assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize('kernel', [
        None,  # Default kernel
        {'sigma': 0.5, 'trainable': False},  # pass kernels as GaussianRBF objects, with default sigma_median fn
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_contextmmddrift(data, kernel, backend, tmp_path, seed):
    """
    Test ContextMMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_simple'
#    assert cd._detector.x_kernel.sigma == cd_load._detector.x_kernel.sigma
    assert cd._detector.c_kernel.sigma == cd_load._detector.c_kernel.sigma
    assert cd._detector.x_kernel.init_sigma_fn == cd_load._detector.x_kernel.init_sigma_fn
    assert cd._detector.c_kernel.init_sigma_fn == cd_load._detector.c_kernel.init_sigma_fn
    assert preds['data']['distance'] == pytest.approx(preds_load['data']['distance'], abs=1e-6)
    assert preds['data']['p_val'] == pytest.approx(preds_load['data']['p_val'], abs=1e-6)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_classifieruncertaintydrift(data, classifier, backend, tmp_path, seed):
    """ Test ClassifierDrift on continuous datasets."""
    # Init detector and predict
    X_ref, X_h0 = data
    with fixed_seed(seed):
        cd = ClassifierUncertaintyDrift(X_ref,
                                        model=classifier,
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
@parametrize('regressor', [encoder_model])
def test_save_regressoruncertaintydrift(data, regressor, backend, tmp_path, seed):
    """ Test RegressorDrift on continuous datasets."""
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
def test_save_onlinemmddrift(data, kernel, preprocess_custom, backend, tmp_path, seed):
    """
    Test MMDDriftOnline on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = MMDDriftOnline(X_ref,
                            ert=ERT,
                            backend=backend,
                            preprocess_fn=preprocess_custom,
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
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.n_bootstraps == N_BOOTSTRAPS
    assert cd_load._detector.ert == ERT
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert cd._detector.kernel.sigma == cd_load._detector.kernel.sigma
    assert cd._detector.kernel.init_sigma_fn == cd_load._detector.kernel.init_sigma_fn
    np.testing.assert_array_equal(stats, stats_load)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_onlinelsdddrift(data, preprocess_custom, backend, tmp_path, seed):
    """
    Test LSDDDriftOnline on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = LSDDDriftOnline(X_ref,
                             ert=ERT,
                             backend=backend,
                             preprocess_fn=preprocess_custom,
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
    np.testing.assert_array_almost_equal(preprocess_custom(X_ref), cd_load.get_config()['x_ref'], 5)
    assert cd_load._detector.n_bootstraps == N_BOOTSTRAPS
    assert cd_load._detector.ert == ERT
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    assert cd._detector.kernel.sigma == cd_load._detector.kernel.sigma
    assert cd._detector.kernel.init_sigma_fn == cd_load._detector.kernel.init_sigma_fn
    np.testing.assert_array_equal(stats, stats_load)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_onlinecvmdrift(data, preprocess_custom, tmp_path, seed):
    """
    Test CVMDriftOnline on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Init detector and make predictions
    X_ref, X_h0 = data

    with fixed_seed(seed):
        cd = CVMDriftOnline(X_ref,
                            ert=ERT,
                            preprocess_fn=preprocess_custom,
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
    np.testing.assert_array_almost_equal(preprocess_custom(X_ref), cd_load.get_config()['x_ref'], 5)
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
def test_save_kernel(kernel, backend, tmp_path):
    """
    Unit test for _save/_load_kernel_config, when kernel is a GaussianRBF kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Save kernel to config
    filepath = tmp_path
    filename = Path('mykernel')
    cfg_kernel = _save_kernel_config(kernel, filepath, filename)
    KernelConfig(**cfg_kernel)  # Passing through the pydantic validator gives a degree of testing
    if kernel.__class__.__name__ == 'GaussianRBF':
        assert cfg_kernel['src'] == '@utils.' + backend + '.kernels.GaussianRBF'
    else:
        assert Path(cfg_kernel['src']).suffix == '.dill'
    assert cfg_kernel['trainable'] == kernel.trainable
    if not kernel.trainable and cfg_kernel['sigma'] is not None:
        np.testing.assert_almost_equal(cfg_kernel['sigma'], kernel.sigma, 6)

    # Resolve and load config (_load_kernel_config is called within resolve_config)
    cfg = {'kernel': cfg_kernel, 'backend': backend}
    _prepend_cfg_filepaths(cfg, tmp_path)
    kernel_loaded = resolve_config(cfg, tmp_path)['kernel']

    # Call kernels
    X = np.random.standard_normal((10, 1))
    kernel(X, X)
    kernel_loaded(X, X)

    # Final checks
    assert type(kernel_loaded) == type(kernel)
    np.testing.assert_array_almost_equal(np.array(kernel_loaded.sigma), np.array(kernel.sigma), 5)
    assert kernel_loaded.trainable == kernel.trainable
    assert kernel_loaded.init_sigma_fn == kernel.init_sigma_fn


# `data` passed below as needed in encoder_model, which is used in deep_kernel
@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd)
@parametrize('deep_kernel', [
        {'kernel_a': 'rbf', 'kernel_b': 'rbf', 'eps': 'trainable'},  # Default for kernel_a and kernel_b, trainable eps
        {'kernel_a': {'trainable': True}, 'kernel_b': 'rbf', 'eps': 0.01},  # Explicit kernel_a, fixed eps
    ], indirect=True
)
def test_save_deepkernel(data, deep_kernel, backend, tmp_path):
    """
    Unit test for _save/_load_kernel_config, when kernel is a DeepKernel kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Get data dim
    X, _ = data
    input_dim = X.shape[1]

    # Save kernel to config
    filepath = tmp_path
    filename = 'mykernel'
    cfg_kernel = _save_kernel_config(deep_kernel, filepath, filename)
    cfg_kernel['proj'], _ = _save_model_config(cfg_kernel['proj'], base_path=filepath, input_shape=input_dim,
                                               backend=backend)
    cfg_kernel = _path2str(cfg_kernel)
    cfg_kernel['proj'] = ModelConfig(**cfg_kernel['proj']).dict()  # Pass thru ModelConfig to set `custom_objects` etc
    cfg_kernel = DeepKernelConfig(**cfg_kernel).dict()  # pydantic validation
    assert cfg_kernel['proj']['src'] == 'model'
    assert cfg_kernel['proj']['custom_objects'] is None

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
    assert pytest.approx(deep_kernel.eps.numpy(), abs=1e-4) == kernel_loaded.eps.numpy()
    assert kernel_loaded.kernel_a.sigma == deep_kernel.kernel_a.sigma
    assert kernel_loaded.kernel_b.sigma == deep_kernel.kernel_b.sigma


@parametrize('preprocess_fn', [preprocess_custom, preprocess_hiddenoutput])
@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
def test_save_preprocess(data, preprocess_fn, tmp_path, backend):
    """
    Unit test for _save_preprocess_config and _load_preprocess_config, with continuous data.

    preprocess_fn's are saved (serialized) and then loaded, with assertions to check equivalence.
    Note: _save_model_config, _save_embedding_config, _save_tokenizer_config, _load_model_config,
     _load_embedding_config, _load_tokenizer_config and _prep_model_and_embedding are all well covered by this test.
    """
    # Save preprocess_fn to config
    filepath = tmp_path
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cfg_preprocess = _save_preprocess_config(preprocess_fn,
                                             backend=backend,
                                             input_shape=input_dim,
                                             filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()  # pydantic validation
    assert cfg_preprocess['src'] == '@cd.' + backend + '.preprocess.preprocess_drift'
    assert cfg_preprocess['model']['src'] == 'preprocess_fn/model'
    # TODO - check layer details here once implemented

    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess, 'backend': backend}
    preprocess_fn_load = resolve_config(cfg, tmp_path)['preprocess_fn']  # tests _load_preprocess_config implicitly
    if backend == 'tensorflow':
        assert preprocess_fn_load.func.__name__ == 'preprocess_drift'
        assert isinstance(preprocess_fn_load.keywords['model'], tf.keras.Model)


@parametrize('preprocess_fn', [preprocess_nlp])
@parametrize_with_cases("data", cases=TextData.movie_sentiment_data, prefix='data_')
def test_save_preprocess_nlp(data, preprocess_fn, max_len, tmp_path, backend):
    """
    Unit test for _save_preprocess_config and _load_preprocess_config, with text data.

    Note: _save_model_config, _save_embedding_config, _save_tokenizer_config, _load_model_config,
     _load_embedding_config, _load_tokenizer_config and _prep_model_and_embedding are all covered by this test.
    """
    # Save preprocess_fn to config
    filepath = tmp_path
    cfg_preprocess = _save_preprocess_config(preprocess_fn,
                                             backend=backend,
                                             input_shape=max_len,
                                             filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()  # pydantic validation
    assert cfg_preprocess['src'] == '@cd.' + backend + '.preprocess.preprocess_drift'
    assert cfg_preprocess['embedding']['src'] == 'preprocess_fn/embedding'
    assert cfg_preprocess['tokenizer']['src'] == 'preprocess_fn/tokenizer'

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
        embedding = preprocess_fn.keywords['model']
        embedding_load = preprocess_fn_load.keywords['model']
    else:
        embedding = preprocess_fn.keywords['model'].encoder.layers[0]
        embedding_load = preprocess_fn_load.keywords['model'].encoder.layers[0]
    assert isinstance(embedding_load.model, type(embedding.model))
    assert embedding_load.emb_type == embedding.emb_type
    assert embedding_load.hs_emb.keywords['layers'] == embedding.hs_emb.keywords['layers']


@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
@parametrize('model', [encoder_model])
@parametrize('layer', [None, -1])
def test_save_model(data, model, layer, backend, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config.
    """
    # Save model
    filepath = tmp_path
    input_dim = data[0].shape[1]
    cfg_model, _ = _save_model_config(model, base_path=filepath, input_shape=input_dim, backend=backend)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()
    assert tmp_path.joinpath('model/model.h5').is_file()

    # Adjust config
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here
    if layer is not None:
        cfg_model['layer'] = layer

    # Load model
    model_load = _load_model_config(cfg_model, backend=backend)
    if layer is None:
        assert isinstance(model_load, type(model))
    else:
        assert isinstance(model_load, (HiddenOutput_tf, HiddenOutput_pt))


def test_save_optimizer(backend):
    class_name = 'Adam'
    learning_rate = 0.01
    epsilon = 1e-7
    amsgrad = False

    if backend == 'tensorflow':
        # Load
        cfg_opt = {
            'class_name': class_name,
            'config': {
                'name': class_name,
                'learning_rate': learning_rate,
                'epsilon': epsilon,
                'amsgrad': amsgrad
            }
        }
        optimizer = _load_optimizer_config(cfg_opt, backend=backend)
        assert type(optimizer).__name__ == class_name
        assert optimizer.learning_rate == learning_rate
        assert optimizer.epsilon == epsilon
        assert optimizer.amsgrad == amsgrad

    # TODO - pytorch


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
        assert type(obj) == type(v)


def test_set_dtypes(backend):
    """
    Unit test to test _set_dtypes.
    """
    if backend == 'tensorflow':
        dtype = 'tf.float32'
    elif backend == 'pytorch':
        dtype = 'torch.float32'
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
