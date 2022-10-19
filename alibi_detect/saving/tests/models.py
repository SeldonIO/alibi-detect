from functools import partial

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from requests.exceptions import HTTPError

import pytest
from pytest_cases import fixture, parametrize
from transformers import AutoTokenizer
from alibi_detect.cd.pytorch import UAE as UAE_pt
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_pt
from alibi_detect.cd.tensorflow import UAE as UAE_tf
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.utils.pytorch.kernels import GaussianRBF as GaussianRBF_pt
from alibi_detect.utils.pytorch.kernels import DeepKernel as DeepKernel_pt
from alibi_detect.utils.tensorflow.kernels import GaussianRBF as GaussianRBF_tf
from alibi_detect.utils.tensorflow.kernels import DeepKernel as DeepKernel_tf
from alibi_detect.models.pytorch import TransformerEmbedding as TransformerEmbedding_pt
from alibi_detect.models.tensorflow import TransformerEmbedding as TransformerEmbedding_tf
from alibi_detect.cd.pytorch import HiddenOutput as HiddenOutput_pt
from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf

LATENT_DIM = 2  # Must be less than input_dim set in ./datasets.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    elif backend == 'pytorch':
        model = nn.Sequential(nn.Linear(input_dim, 5),
                              nn.ReLU(),
                              nn.Linear(5, LATENT_DIM))
    else:
        pytest.skip('`encoder_model` only implemented for tensorflow and pytorch.')
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
    elif backend == 'pytorch':
        model = nn.Sequential(nn.Linear(input_dim, 5),
                              nn.ReLU(),
                              nn.Dropout(0.0),  # To ensure determinism
                              nn.Linear(5, LATENT_DIM))
    else:
        pytest.skip('`encoder_dropout_model` only implemented for tensorflow and pytorch.')
    return model


@fixture
def preprocess_custom(encoder_model):
    """
    Preprocess function with Untrained Autoencoder.
    """
    if isinstance(encoder_model, tf.keras.Model):
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
        return None
    elif isinstance(kernel, dict):  # dict of kwargs
        kernel_cfg = kernel.copy()
        sigma = kernel_cfg.pop('sigma', None)
        if backend == 'tensorflow':
            if sigma is not None and not isinstance(sigma, tf.Tensor):
                sigma = tf.convert_to_tensor(sigma)
            kernel = GaussianRBF_tf(sigma=sigma, **kernel_cfg)
        elif backend == 'pytorch':
            if sigma is not None and not isinstance(sigma, torch.Tensor):
                sigma = torch.tensor(sigma)
            kernel = GaussianRBF_pt(sigma=sigma, **kernel_cfg)
        else:
            pytest.skip('`kernel` only implemented for tensorflow and pytorch.')
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
        kernel_a = GaussianRBF_pt(**kernel_a) if isinstance(kernel_a, dict) else kernel_a
        kernel_a = GaussianRBF_pt(**kernel_b) if isinstance(kernel_b, dict) else kernel_b
        deep_kernel = DeepKernel_pt(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)
    else:
        pytest.skip('`deep_kernel` only implemented for tensorflow and pytorch.')
    return deep_kernel


@fixture
def classifier_model(backend, current_cases):
    """
    Classification model with given input dimension and backend.
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']
    if backend == 'tensorflow':
        model = tf.keras.Sequential(
               [
                   tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                   tf.keras.layers.Dense(2, activation=tf.nn.softmax),
               ]
           )
    elif backend == 'pytorch':
        model = nn.Sequential(nn.Linear(input_dim, 2),
                              nn.Softmax(1))
    elif backend == 'sklearn':
        model = RandomForestClassifier()
    else:
        pytest.skip('`classifier_model` only implemented for tensorflow, pytorch, and sklearn.')
    return model


@fixture
def xgb_classifier_model():
    model = XGBClassifier()
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (OSError, HTTPError):
        pytest.skip(f"Problem downloading {model_name} from huggingface.co")
    X = 'A dummy string'  # this will be padded to max_len
    tokens = tokenizer(list(X[:5]), pad_to_max_length=True,
                       max_length=max_len, return_tensors=backend)

    # Load embedding model
    emb_type = 'hidden_state'
    n_layers = 8
    layers = [-_ for _ in range(1, n_layers + 1)]
    enc_dim = 32

    if backend == 'tf':
        try:
            embedding = TransformerEmbedding_tf(model_name, emb_type, layers)
        except (OSError, HTTPError):
            pytest.skip(f"Problem downloading {model_name} from huggingface.co")
        if uae:
            x_emb = embedding(tokens)
            shape = (x_emb.shape[1],)
            embedding = UAE_tf(input_layer=embedding, shape=shape, enc_dim=enc_dim)
    elif backend == 'pt':
        try:
            embedding = TransformerEmbedding_pt(model_name, emb_type, layers)
        except (OSError, HTTPError):
            pytest.skip(f"Problem downloading {model_name} from huggingface.co")
        if uae:
            x_emb = embedding(tokens)
            shape = (x_emb.shape[1],)
            embedding = UAE_pt(input_layer=embedding, shape=shape, enc_dim=enc_dim)

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
    elif backend == 'pytorch':
        preprocess_fn = partial(preprocess_drift_pt, model=embedding, tokenizer=tokenizer, max_len=max_len,
                                preprocess_batch_fn=preprocess_simple)
    else:
        pytest.skip('`preprocess_nlp` only implemented for tensorflow and pytorch.')
    return preprocess_fn


@fixture
def preprocess_hiddenoutput(classifier_model, current_cases, backend):
    """
    Preprocess function to extract the softmax layer of a classifier (with the HiddenOutput utility function).
    """
    _, _, data_params = current_cases["data"]
    _, input_dim = data_params['data_shape']

    if backend == 'tensorflow':
        model = HiddenOutput_tf(classifier_model, layer=-1, input_shape=(None, input_dim))
        preprocess_fn = partial(preprocess_drift_tf, model=model)
    elif backend == 'pytorch':
        model = HiddenOutput_pt(classifier_model, layer=-1)
        preprocess_fn = partial(preprocess_drift_pt, model=model)
    else:
        pytest.skip('`preprocess_hiddenoutput` only implemented for tensorflow and pytorch.')
    return preprocess_fn
