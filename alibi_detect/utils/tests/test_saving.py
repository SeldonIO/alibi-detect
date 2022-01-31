# type: ignore
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
from alibi_detect.utils.saving import (save_detector, _save_kernel, _save_preprocess, _save_embedding,
                                       _save_tokenizer, _save_model, _resolve_paths)  # type: ignore
from alibi_detect.utils.loading import (load_detector, _load_kernel, resolve_cfg, _load_preprocess, _load_embedding,
                                        _load_tokenizer, _load_model)  # type: ignore
from alibi_detect.utils.schemas import (
    KernelConfig, KernelConfigResolved,
    DeepKernelConfig, DeepKernelConfigResolved,
    PreprocessConfig, PreprocessConfigResolved,
    EmbeddingConfig
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

#TODO: Some of the fixtures can/should be moved elsewhere (i.e. if they can be recycled for use elsewhere)


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


@fixture
def deep_kernel(backend, current_cases):
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


##@fixture
#def


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
    # assert det.predict(X_ref)['data']['p_val'] == det_load.predict(X_ref)['data']['p_val']
    # Commented as settings tf/np seeds does not currently make deterministic


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
    cfg_kernel = _save_kernel(kernel, filepath, device=DEVICE, filename=filename, verbose=False)
    KernelConfig(**cfg_kernel).dict()
    # TODO assertions

    # Resolve and load config
    cfg = {'kernel': cfg_kernel}
    cfg_kernel = _resolve_paths(resolve_cfg(cfg, tmp_path)['kernel'])
    cfg_kernel = KernelConfigResolved(**cfg_kernel).dict()
    kernel_loaded = _load_kernel(cfg_kernel, backend=backend, device=DEVICE)
    assert type(kernel_loaded) == type(kernel)
    np.testing.assert_array_equal(np.array(kernel_loaded.sigma), np.array(cfg_kernel['sigma']))
    assert kernel_loaded.trainable == cfg_kernel['trainable']


def test_save_deepkernel(deep_kernel, backend, tmp_path):
    """
    Unit test for _save/_load_kernel, when kernel is a DeepKernel kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # Prep cfg_kernel (detector's .get_config() would usually be doing this)
    cfg_kernel = {
        'proj': deep_kernel.proj,
        'eps': deep_kernel.eps,
        'kernel_a': deep_kernel.kernel_a,
        'kernel_b': deep_kernel.kernel_b
    }
    # Save kernel to config
    filepath = tmp_path
    filename = 'mykernel.dill'
    cfg_kernel = _save_kernel(cfg_kernel, filepath, device=DEVICE, filename=filename, verbose=False)
    print(cfg_kernel)
    DeepKernelConfig(**cfg_kernel).dict()
    # TODO add more assertions here

    # Resolve and load config
    cfg = {'kernel': cfg_kernel}
    cfg_kernel = _resolve_paths(resolve_cfg(cfg, tmp_path)['kernel'])
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
                                           filepath=filepath,
                                           verbose=False)
    cfg_preprocess = _resolve_paths(cfg_preprocess)
    PreprocessConfig(**cfg_preprocess).dict()
    # TODO - assertions to test cfg_preprocess

    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess}
    cfg_preprocess = resolve_cfg(cfg, tmp_path)['preprocess_fn']
    cfg_preprocess = PreprocessConfigResolved(**cfg_preprocess).dict()
    preprocess_fn_load = _load_preprocess(cfg_preprocess, backend, verbose=False)
    # TODO - post loading assertions


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
                                           filepath=filepath,
                                           verbose=False)
    cfg_preprocess = _resolve_paths(cfg_preprocess)
    PreprocessConfig(**cfg_preprocess).dict()
#    # TODO - assertions to test cfg_preprocess
#
    # Resolve and load preprocess config
    cfg = {'preprocess_fn': cfg_preprocess}
    cfg_preprocess = resolve_cfg(cfg, tmp_path)['preprocess_fn']
    preprocess_fn_load = _load_preprocess(cfg_preprocess, backend, verbose=False)
    tokenizer = preprocess_fn_load.keywords['tokenizer']
    embedding = preprocess_fn_load.keywords['model']
#    assert tokenizer...
    # TODO - post loading assertions


# TODO - test loading of UAE and HiddenOutput etc? Pending further discussion (no UAE for pytorch etc)
#def test_save_model():
#    pass


## TODO- unit tests
##       - save/load_model
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