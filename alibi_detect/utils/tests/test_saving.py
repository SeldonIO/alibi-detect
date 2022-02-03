# type: ignore
"""
Tests for saving/loading of detectors via config.toml files.

Internal functions such as save_kernel/load_kernel etc are also tested.
"""
# TODO - test pytorch save/load functionality
from functools import partial

import dill
import numpy as np
import scipy
import pytest
import random
from pathlib import Path
from pytest_cases import parametrize_with_cases, parametrize, fixture, param_fixture
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import torch
from typing import Callable
from packaging import version
from transformers import AutoTokenizer
from alibi_detect.models.tensorflow import TransformerEmbedding as TransformerEmbedding_tf
from alibi_detect.models.pytorch import TransformerEmbedding as TransformerEmbedding_pt
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf, UAE as UAE_tf
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_pt
from alibi_detect.utils.tensorflow.kernels import DeepKernel as DeepKernel_tf, GaussianRBF as GaussianRBF_tf
from alibi_detect.utils.pytorch.kernels import DeepKernel as DeepKernel_pt, GaussianRBF as GaussianRBF_pt
from alibi_detect.utils.registry import registry
from alibi_detect.utils.saving import (save_detector, _save_kernel, _save_preprocess,
                                       _save_model, _path2str, _serialize_function)  # type: ignore
from alibi_detect.utils.loading import (load_detector, _load_kernel, resolve_cfg, _load_preprocess,
                                        _load_model, _load_optimizer, _set_nested_value, _replace,
                                        _get_nested_value)  # type: ignore
from alibi_detect.utils.schemas import (
    KernelConfig, KernelConfigResolved,
    DeepKernelConfig, DeepKernelConfigResolved,
    PreprocessConfig, PreprocessConfigResolved,
    ModelConfig
)
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
from datasets import ContinuousData, BinData, CategoricalData, MixedData, TextData
if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift

backend = param_fixture("backend", ['tensorflow'])
P_VAL = 0.05
N_PERMUTATIONS = 10
LATENT_DIM = 2  # Must be less than input_dim set in ./datasets.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REGISTERED_OBJECTS = registry.get_all()

# Set seeds (TODO - may also need to set env variables when using cuda)
np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
random.seed(0)

#  TODO: Some of the fixtures can/should be moved elsewhere (i.e. if they can be recycled for use elsewhere)


@fixture
def uae_model(backend, current_cases):
    """
    Untrained Autoencoder of given input dimension and backend.
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
def preprocess_uae(uae_model, backend):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if backend == 'tensorflow':
        preprocess_fn = partial(preprocess_drift_tf, model=uae_model)
    else:
        preprocess_fn = partial(preprocess_drift_pt, model=uae_model)
    return preprocess_fn


@fixture
def kernel(request, backend):
    """
    Gaussian RBF kernel for given backend.
    """
    sigma = request.param['sigma']
    trainable = request.param['trainable']
    if backend == 'tensorflow':
        kernel = GaussianRBF_tf(sigma=sigma, trainable=trainable)
    elif backend == 'pytorch':
        kernel = GaussianRBF_pt(sigma=sigma, trainable=trainable)
    else:
        raise ValueError('preprocess_uae `backend` not valid.')
    return kernel


@fixture(unpack_into=('deep_kernel, kernel_proj_dim'))
def build_deep_kernel(backend, current_cases):
    """
    Deep kernel with given input dimension and backend.
    """
    if 'data' in current_cases:
        _, _, data_params = current_cases['data']
        _, input_dim = data_params['data_shape']
    else:
        input_dim = 4

    if backend == 'tensorflow':
        proj = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer((1, 1, input_dim,)),
              tf.keras.layers.Conv1D(int(input_dim), 2, strides=1, padding='same', activation=tf.nn.relu),
              tf.keras.layers.Conv1D(input_dim, 2, strides=1, padding='same', activation=tf.nn.relu),
              tf.keras.layers.Flatten(),
          ]
        )
        deep_kernel = DeepKernel_tf(proj, eps=0.01)
    elif backend == 'pytorch':
        raise NotImplementedError('`pytorch` tests not implemented.')
    else:
        raise ValueError('preprocess_uae `backend` not valid.')
    return deep_kernel, input_dim


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
        raise ValueError('preprocess_uae `backend` not valid.')
    return model


@fixture(unpack_into=('tokenizer, embedding, enc_dim'))
@parametrize('model_name, max_len', [('bert-base-cased', 100)])
def nlp_embedding_and_tokenizer(model_name, max_len, backend):
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
        x_emb = embedding(tokens)
        shape = (x_emb.shape[1],)
        embedding = UAE_tf(input_layer=embedding, shape=shape, enc_dim=enc_dim)
    else:
        embedding = TransformerEmbedding_pt(model_name, emb_type, layers)
        x_emb = embedding(tokens)
        emb_dim = x_emb.shape[1]
        device = torch.device(DEVICE)
        embedding = torch.nn.Sequential(
            embedding,
            torch.nn.Linear(emb_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, enc_dim)
        ).to(device).eval()

    return tokenizer, embedding, enc_dim


@fixture
def preprocess_nlp(embedding, tokenizer, current_cases, backend):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if backend == 'tensorflow':
        preprocess_fn = partial(preprocess_drift_tf, model=embedding, tokenizer=tokenizer)
    else:
        preprocess_fn = partial(preprocess_drift_pt, model=embedding, tokenizer=tokenizer)
    return preprocess_fn


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_ksdrift(data, preprocess_uae, tmp_path):
    """
    Test KSDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
    assert cd_load.x_ref_preprocessed
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_mmddrift(data, preprocess_uae, backend, tmp_path):
    """
    Test MMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
#    assert cd.predict(X_h0)['data']['p_val'] == cd_load.predict(X_h0)['data']['p_val']  # Not deterministic


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_lsdddrift(data, preprocess_uae, backend, tmp_path):
    """
    Test LSDDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    cd = LSDDDrift(X_ref,
                   p_val=P_VAL,
                   backend=backend,
                   preprocess_fn=preprocess_uae,
                   preprocess_at_init=True,
                   n_permutations=N_PERMUTATIONS,
                   )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # assertions
    np.testing.assert_almost_equal(cd._detector._normalize(preprocess_uae(X_ref)), cd_load._detector.x_ref, 10)
    assert cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
#    assert cd.predict(X_h0)['data']['p_val'] == cd_load.predict(X_h0)['data']['p_val']  # Not deterministic


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
    """
    Test SpotTheDiffDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
    """
    Test LearnedKernelDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
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
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.kernel, DeepKernel_tf)
    else:
        assert isinstance(cd_load._detector.kernel, DeepKernel_pt)

# TODO - checks for modeluncertainty detectors


@parametrize('kernel', [
        {'sigma': 0.5, 'trainable': False},
        {'sigma': [0.5, 0.8], 'trainable': False},
        {'sigma': None, 'trainable': True}
    ], indirect=True
)
def test_save_kernel(kernel, backend, tmp_path):
    """
    Unit test for _save/_load_kernel, when kernel is a GaussianRBF kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Save kernel to config
    filepath = tmp_path
    filename = 'mykernel.dill'
    cfg_kernel = _save_kernel(kernel, filepath, device=DEVICE, filename=filename)
    KernelConfig(**cfg_kernel).dict()
    # TODO assertions

    # Resolve and load config
    cfg = {'kernel': cfg_kernel}
    cfg_kernel = _path2str(resolve_cfg(cfg, tmp_path)['kernel'])
    cfg_kernel = KernelConfigResolved(**cfg_kernel).dict()
    kernel_loaded = _load_kernel(cfg_kernel, backend=backend, device=DEVICE)
    assert type(kernel_loaded) == type(kernel)
    np.testing.assert_array_equal(np.array(kernel_loaded.sigma), np.array(cfg_kernel['sigma']))
    assert kernel_loaded.trainable == cfg_kernel['trainable']


def test_save_deepkernel(deep_kernel, kernel_proj_dim, backend, tmp_path):
    """
    Unit test for _save/_load_kernel, when kernel is a DeepKernel kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Prep cfg_kernel (detector's .get_config() would usually be doing this)
    cfg_kernel = {
        'proj': deep_kernel.proj,
        'eps': deep_kernel.eps.numpy(),
        'kernel_a': deep_kernel.kernel_a,
        'kernel_b': deep_kernel.kernel_b
    }
    # Save kernel to config
    filepath = tmp_path
    filename = 'mykernel.dill'
    cfg_kernel = _save_kernel(cfg_kernel, filepath, device=DEVICE, filename=filename)
    cfg_kernel['proj'], _ = _save_model(cfg_kernel['proj'], base_path=filepath, input_shape=kernel_proj_dim,
                                        backend=backend)
    cfg_kernel = _path2str(cfg_kernel)
    cfg_kernel['proj'] = ModelConfig(**cfg_kernel['proj']).dict()  # Pass through ModelConfig to set `custom_obj` etc
    cfg_kernel = DeepKernelConfig(**cfg_kernel).dict()
    # TODO add more assertions here

    # Resolve and load config
    cfg = {'kernel': cfg_kernel}
    cfg_kernel = resolve_cfg(cfg, tmp_path)['kernel']
    cfg_kernel = DeepKernelConfigResolved(**cfg_kernel).dict()
    kernel_loaded = _load_kernel(cfg_kernel, backend=backend, device=DEVICE)
    assert isinstance(kernel_loaded.proj, (torch.nn.Module, tf.keras.Model))
    np.testing.assert_almost_equal(kernel_loaded.eps, deep_kernel.eps, 4)
    assert kernel_loaded.kernel_a.sigma == deep_kernel.kernel_a.sigma
    assert kernel_loaded.kernel_b.sigma == deep_kernel.kernel_b.sigma


@parametrize('preprocess_fn', [preprocess_uae])
@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
def test_save_preprocess(data, preprocess_fn, tmp_path, backend):
    """
    Unit test for _save_preprocess and _load_preprocess, with continuous data.

    preprocess_fn's are saved (serialized) and then loaded, with assertions to check equivalence.
    Note: _save_model, _save_embedding, _save_tokenizer, _load_model, _load_embedding, _load_tokenizer and
     _prep_model_and_embedding are all well covered by this test.
    """
    # Save preprocess_fn to config
    filepath = tmp_path
    X_ref, X_h0 = data
    input_dim = X_ref.shape[1]
    cfg_preprocess = _save_preprocess(preprocess_fn,
                                      backend=backend,
                                      input_shape=input_dim,
                                      filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()
    # TODO - assertions to test cfg_preprocess

    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess}
    cfg_preprocess = resolve_cfg(cfg, tmp_path)['preprocess_fn']
    cfg_preprocess = PreprocessConfigResolved(**cfg_preprocess).dict()
    preprocess_fn_load = _load_preprocess(cfg_preprocess, backend)
    if backend == 'tensorflow':
        assert isinstance(preprocess_fn_load.keywords['model'], UAE_tf)
        # NOTE: can't currently compare to original as loaded model wrapped in UAE. See note in loading.py
    # TODO - more post loading assertions


@parametrize('preprocess_fn', [preprocess_nlp])
@parametrize_with_cases("data", cases=TextData.movie_sentiment_data, prefix='data_')
def test_save_preprocess_nlp(data, preprocess_fn, enc_dim, tmp_path, backend):
    """
    Unit test for _save_preprocess and _load_preprocess, with text data.

    Note: _save_model, _save_embedding, _save_tokenizer, _load_model, _load_embedding, _load_tokenizer and
     _prep_model_and_embedding are all covered by this test.
    """
    # Save preprocess_fn to config
    filepath = tmp_path
    cfg_preprocess = _save_preprocess(preprocess_fn,
                                      backend=backend,
                                      input_shape=enc_dim,
                                      filepath=filepath)
    cfg_preprocess = _path2str(cfg_preprocess)
    cfg_preprocess = PreprocessConfig(**cfg_preprocess).dict()
#    # TODO - assertions to test cfg_preprocess
#
    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess}
    cfg_preprocess = resolve_cfg(cfg, tmp_path)['preprocess_fn']
    preprocess_fn_load = _load_preprocess(cfg_preprocess, backend)
    assert isinstance(preprocess_fn_load.keywords['tokenizer'], type(preprocess_fn.keywords['tokenizer']))
    assert isinstance(preprocess_fn_load.keywords['model'], type(preprocess_fn.keywords['model']))


# TODO - test loading of UAE and HiddenOutput etc? Pending further discussion (no UAE for pytorch etc)
@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
@parametrize('model', [uae_model])
def test_save_model(data, model, backend, tmp_path):
    """
    Unit test for _save_model and _load_model.
    """
    # Save model
    filepath = tmp_path
    input_dim = data[0].shape[1]
    cfg_model, _ = _save_model(model, base_path=filepath, input_shape=input_dim, backend=backend)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()
    assert tmp_path.joinpath('model/model.h5').is_file()

    # Load model
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here
    model_load = _load_model(cfg_model, backend=backend)
    assert isinstance(model_load, type(model))
    # TODO - double check why loaded model is Sequential but UAE in test_save_preprocess
    # TODO - Assertions should depend on cfg_model['type'] when more models are parametrized


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
        optimizer = _load_optimizer(cfg_opt, backend=backend)
        print(optimizer)
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
    assert rel_path == str(tmp_path)

    cfg_abs = _path2str(cfg, absolute=True)
    abs_path = cfg_abs['dict']['a path']
    assert isinstance(abs_path, str)
    assert abs_path == str(tmp_path.resolve())


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
    src, kwargs = _serialize_function(partial_func, base_path=tmp_path, local_path=Path('function'))
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
    src, kwargs = _serialize_function(function, base_path=tmp_path)
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


# TODO - could do with a final test to check resolve_cfg works with all registered functions, file types etc.
#  wait until after design finalised!
