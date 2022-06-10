import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import torch
import torch.nn as nn
from alibi_detect.cd import ClassifierDrift
from alibi_detect.cd.pytorch.classifier import ClassifierDriftTorch
from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF
from alibi_detect.cd.sklearn.classifier import ClassifierDriftSklearn
from sklearn.neural_network import MLPClassifier
from typing import Tuple

n, n_features = 100, 5


def tensorflow_model(input_shape: Tuple[int]):
    x_in = Input(shape=input_shape)
    x = Dense(20, activation=tf.nn.relu)(x_in)
    x_out = Dense(2, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


def pytorch_model(input_shape: int):
    return torch.nn.Sequential(
        nn.Linear(input_shape, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )


def sklearn_model():
    return MLPClassifier(hidden_layer_sizes=(20, ))


tests_clfdrift = ['tensorflow', 'pytorch', 'PyToRcH', 'sklearn', 'mxnet']
n_tests = len(tests_clfdrift)


@pytest.fixture
def clfdrift_params(request):
    return tests_clfdrift[request.param]


@pytest.mark.parametrize('clfdrift_params', list(range(n_tests)), indirect=True)
def test_clfdrift(clfdrift_params):
    backend = clfdrift_params
    if backend.lower() == 'pytorch':
        model = pytorch_model(n_features)
    elif backend.lower() == 'tensorflow':
        model = tensorflow_model((n_features,))
    elif backend.lower() == 'sklearn':
        model = sklearn_model()
    else:
        model = None
    x_ref = np.random.randn(*(n, n_features))

    try:
        cd = ClassifierDrift(x_ref=x_ref, model=model, backend=backend)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, ClassifierDriftTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, ClassifierDriftTF)
    elif backend.lower() == 'sklearn':
        assert isinstance(cd._detector, ClassifierDriftSklearn)
    else:
        assert cd is None
