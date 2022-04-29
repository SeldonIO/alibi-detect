# type: ignore
"""
Tests for saving/loading of detectors via config.toml files.

Internal functions such as save_kernel/load_kernel_config etc are also tested.
"""
import random
# TODO future - test pytorch save/load functionality
# TODO (could/should also add tests to backend-specific submodules)
from functools import partial
from pathlib import Path
from typing import Callable

import dill
import numpy as np
import pytest
import scipy
import tensorflow as tf
import torch

import alibi_detect.saving.registry
from datasets import (BinData, CategoricalData, ContinuousData, MixedData,
                      TextData)
from packaging import version
from pytest_cases import (fixture, param_fixture, parametrize,
                          parametrize_with_cases)
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

from alibi_detect.cd import (ChiSquareDrift,  # ClassifierUncertaintyDrift,
                             ClassifierDrift, FETDrift, KSDrift,
                             LearnedKernelDrift, LSDDDrift, MMDDrift,
                             SpotTheDiffDrift, TabularDrift, ContextMMDDrift)
from alibi_detect.cd.pytorch import HiddenOutput as HiddenOutput_pt
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_pt
from alibi_detect.cd.tensorflow import UAE as UAE_tf
from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.models.pytorch import \
    TransformerEmbedding as TransformerEmbedding_pt
from alibi_detect.models.tensorflow import \
    TransformerEmbedding as TransformerEmbedding_tf
from alibi_detect.saving import (load_detector, read_config, registry,
                                 resolve_config, save_detector, write_config)
from alibi_detect.saving.loading import (_get_nested_value,  # type: ignore
                                         _load_kernel_config,
                                         _load_model_config,
                                         _load_optimizer_config,
                                         _load_preprocess_config, _replace,
                                         _set_dtypes, _set_nested_value, _prepend_cfg_filepaths)
from alibi_detect.saving.saving import _serialize_object  # type: ignore
from alibi_detect.saving.saving import (_path2str, _save_kernel_config,
                                        _save_model_config,
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
N_PERMUTATIONS = 10
LATENT_DIM = 2  # Must be less than input_dim set in ./datasets.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REGISTERED_OBJECTS = registry.get_all()

# Set seeds
np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
random.seed(0)

# TODO - future: Some of the fixtures can/should be moved elsewhere (i.e. if they can be recycled for use elsewhere)


@fixture
def custom_model(backend, current_cases):
    """
    An untrained autoencoder of given input dimension and backend (this is a "custom" model, NOT an Alibi Detect UAE).
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
def preprocess_custom(custom_model, backend):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if backend == 'tensorflow':
        preprocess_fn = partial(preprocess_drift_tf, model=custom_model)
    else:
        preprocess_fn = partial(preprocess_drift_pt, model=custom_model)
    return preprocess_fn


@fixture
def kernel(request, backend):
    """
    Gaussian RBF kernel for given backend.
    """
    kernel_kwargs = request.param
    if backend == 'tensorflow':
        kernel = GaussianRBF_tf(**kernel_kwargs) if kernel_kwargs is not None else GaussianRBF_tf
    elif backend == 'pytorch':
        kernel = GaussianRBF_pt(**kernel_kwargs) if kernel_kwargs is not None else GaussianRBF_pt
    else:
        raise ValueError('`backend` not valid.')
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
        raise ValueError('`backend` not valid.')
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


@fixture
def preprocess_batch(x: np.ndarray):
    """
    Dummy function to test serialization of generic Python function within preprocess_fn.
    """
    assert isinstance(x, np.ndarray)
    return x


@fixture
def preprocess_nlp(embedding, tokenizer, max_len, backend):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if backend == 'tensorflow':
        preprocess_fn = partial(preprocess_drift_tf, model=embedding, tokenizer=tokenizer,
                                max_len=max_len, preprocess_batch_fn=preprocess_batch)
    else:
        preprocess_fn = partial(preprocess_drift_pt, model=embedding, tokenizer=tokenizer, max_len=max_len,
                                preprocess_batch_fn=preprocess_batch)
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
    assert cd_load.x_ref_preprocessed
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
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
    assert cd_load.x_ref_preprocessed
    if isinstance(preprocess_fn.keywords['model'], (TransformerEmbedding_tf, TransformerEmbedding_pt)):
        assert cd_load.n_features == 768  # hardcoded to bert-base-cased for now
    else:
        assert cd_load.n_features == enc_dim  # encoder dim
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
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
    assert cd_load.x_ref_preprocessed
    assert cd_load.n_features == LATENT_DIM
    assert cd_load.p_val == P_VAL
    assert isinstance(cd_load.preprocess_fn, Callable)
    assert cd_load.preprocess_fn.func.__name__ == 'preprocess_drift'
    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_mmddrift(data, preprocess_custom, backend, tmp_path):
    """
    Test MMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    cd = MMDDrift(X_ref,
                  p_val=P_VAL,
                  backend=backend,
                  preprocess_fn=preprocess_custom,
                  n_permutations=N_PERMUTATIONS,
                  preprocess_at_init=True,
                  )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # assertions
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.x_ref_preprocessed
    assert not cd_load._detector.infer_sigma
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
#    assert cd.predict(X_h0)['data']['p_val'] == cd_load.predict(X_h0)['data']['p_val']  # Not deterministic


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_lsdddrift(data, preprocess_custom, backend, tmp_path):
    """
    Test LSDDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    cd = LSDDDrift(X_ref,
                   p_val=P_VAL,
                   backend=backend,
                   preprocess_fn=preprocess_custom,
                   preprocess_at_init=True,
                   n_permutations=N_PERMUTATIONS,
                   )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # assertions
    np.testing.assert_almost_equal(cd._detector._normalize(preprocess_custom(X_ref)), cd_load._detector.x_ref, 10)
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
    assert cd_load.x_ref_categories == cd.x_ref_categories


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_classifierdrift(data, preprocess_custom, classifier, backend, tmp_path):
    """ Test ClassifierDrift on continuous datasets."""
    # Detector save/load
    X_ref, X_h0 = data
    cd = ClassifierDrift(X_ref,
                         model=classifier,
                         p_val=P_VAL,
                         preprocess_fn=preprocess_custom,
                         n_folds=5,
                         backend=backend,
                         train_size=None)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector.x_ref)
    assert isinstance(cd_load._detector.skf, StratifiedKFold)
    assert cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.model, tf.keras.Model)
    else:
        pass  # TODO
#    np.testing.assert_array_equal(cd.predict(X_h0)['data']['p_val'],  # only do for deterministic detectors
#                                  cd_load.predict(X_h0)['data']['p_val'])


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_spotthediff(data, classifier, preprocess_custom, backend, tmp_path):
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
                          preprocess_fn=preprocess_custom)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector._detector.x_ref)
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
def test_save_learnedkernel(data, deep_kernel, preprocess_custom, backend, tmp_path):
    """
    Test LearnedKernelDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    cd = LearnedKernelDrift(X_ref,
                            deep_kernel,
                            p_val=P_VAL,
                            backend=backend,
                            preprocess_fn=preprocess_custom,
                            train_size=0.7)
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # Assert
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector.x_ref)
    assert cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.train_kwargs, dict)
    if backend == 'tensorflow':
        assert isinstance(cd_load._detector.kernel, DeepKernel_tf)
    else:
        assert isinstance(cd_load._detector.kernel, DeepKernel_pt)

# TODO - checks for modeluncertainty detectors - once save/load implemented for them

@parametrize('kernel', [
        None,  # detector's default kernels
        {'sigma': 0.5, 'trainable': False},  # custom kernels
    ], indirect=True
)
@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
def test_save_contextmmddrift(data, preprocess_custom, kernel, backend, tmp_path):
    """
    Test ContextMMDDrift on continuous datasets, with UAE as preprocess_fn.

    Detector is saved and then loaded, with assertions checking that the reinstantiated detector is equivalent.
    """
    # Detector save/load
    X_ref, X_h0 = data
    C_ref = X_ref[:, 0] + 1
    cd = ContextMMDDrift(X_ref,
                         C_ref,
                         p_val=P_VAL,
                         backend=backend,
                         preprocess_fn=preprocess_custom,
                         n_permutations=N_PERMUTATIONS,
                         preprocess_at_init=True,
                         x_kernel=kernel,
                         c_kernel=kernel
                         )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    # assertions
    np.testing.assert_array_equal(preprocess_custom(X_ref), cd_load._detector.x_ref)
    np.testing.assert_array_equal(C_ref, cd_load._detector.c_ref)
    assert cd_load._detector.x_ref_preprocessed
    assert cd_load._detector.n_permutations == N_PERMUTATIONS
    assert cd_load._detector.p_val == P_VAL
    assert isinstance(cd_load._detector.preprocess_fn, Callable)
    assert cd_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
    print(cd._detector.x_kernel.sigma, cd_load._detector.x_kernel.sigma)
    print(cd._detector.x_kernel.init_sigma_fn, cd_load._detector.x_kernel.init_sigma_fn)
    assert cd._detector.x_kernel.sigma == cd_load._detector.x_kernel.sigma
    assert cd._detector.x_kernel.init_sigma_fn == cd_load._detector.x_kernel.init_sigma_fn
#    assert cd.predict(X_h0)['data']['p_val'] == cd_load.predict(X_h0)['data']['p_val']  # Not deterministic


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
    cfg['x_ref'] = x_ref_path
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
@parametrize('use_register', [True, False])
def test_save_kernel(kernel, use_register, backend, tmp_path):
    """
    Unit test for _save/_load_kernel_config, when kernel is a GaussianRBF kernel.

    Kernels are saved and then loaded, with assertions to check equivalence.
    """
    # If use_register False, overwrite GaussianRBF's from object registry to force kernel to be saved to disk
    if not use_register:
        registry.register('utils.' + backend + '.kernels.GaussianRBF', func="NULL")

    # Save kernel to config
    filepath = tmp_path
    filename = 'mykernel'
    cfg_kernel = _save_kernel_config(kernel, filepath, filename)
    KernelConfig(**cfg_kernel)  # Passing through the pydantic validator gives a degree of testing
    if use_register:
        assert cfg_kernel['src'] == '@utils.' + backend + '.kernels.GaussianRBF'
    else:
        assert Path(cfg_kernel['src']).suffix == '.dill'
    assert cfg_kernel['trainable'] == kernel.trainable
    if not kernel.trainable:
        np.testing.assert_almost_equal(cfg_kernel['sigma'], kernel.sigma, 6)

    # Resolve and load config (_load_kernel_config is called within resolve_config)
    cfg = {'kernel': cfg_kernel, 'backend': backend}
    _prepend_cfg_filepaths(cfg, tmp_path)
    kernel_loaded = resolve_config(cfg, tmp_path)['kernel']
    assert type(kernel_loaded) == type(kernel)
    np.testing.assert_array_almost_equal(np.array(kernel_loaded.sigma), np.array(kernel.sigma), -5)
    assert kernel_loaded.trainable == kernel.trainable
    assert kernel_loaded.init_sigma_fn == kernel.init_sigma_fn


def test_save_deepkernel(deep_kernel, kernel_proj_dim, backend, tmp_path):
    """
    Unit test for _save/_load_kernel_config, when kernel is a DeepKernel kernel.

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
    cfg_kernel = _save_kernel_config(cfg_kernel, filepath, filename)
    cfg_kernel['proj'], _ = _save_model_config(cfg_kernel['proj'], base_path=filepath, input_shape=kernel_proj_dim,
                                               backend=backend)
    cfg_kernel = _path2str(cfg_kernel)
    cfg_kernel['proj'] = ModelConfig(**cfg_kernel['proj']).dict()  # Pass thru ModelConfig to set `custom_objects` etc
    cfg_kernel = DeepKernelConfig(**cfg_kernel).dict()  # pydantic validation
    assert cfg_kernel['proj']['src'] == 'model'
    assert cfg_kernel['proj']['custom_objects'] is None
    assert pytest.approx(cfg_kernel['eps'], deep_kernel.eps, 4)
    assert cfg_kernel['kernel_a']['trainable'] and cfg_kernel['kernel_b']['trainable']

    # Resolve and load config
    cfg = {'kernel': cfg_kernel}
    cfg_kernel = resolve_config(cfg, tmp_path)['kernel']
    kernel_loaded = _load_kernel_config(cfg_kernel, backend=backend)
    assert isinstance(kernel_loaded.proj, (torch.nn.Module, tf.keras.Model))
    np.testing.assert_almost_equal(kernel_loaded.eps, deep_kernel.eps, 4)
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
    cfg = {'preprocess_fn': cfg_preprocess}
    cfg_preprocess = resolve_config(cfg, tmp_path)['preprocess_fn']
    preprocess_fn_load = _load_preprocess_config(cfg_preprocess, backend)
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
    cfg = {'preprocess_fn': cfg_preprocess}
    cfg_preprocess = resolve_config(cfg, tmp_path)['preprocess_fn']
    preprocess_fn_load = _load_preprocess_config(cfg_preprocess, backend)
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
@parametrize('model', [custom_model])
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
