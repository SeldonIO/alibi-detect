import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import torch
import torch.nn as nn
from alibi_detect.cd import ClassifierDrift
from alibi_detect.cd.pytorch.classifier import ClassifierDriftTorch
from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF

n, n_features = 100, 5


def mymodel(shape):
    x_in = Input(shape=shape)
    x = Dense(20, activation=tf.nn.relu)(x_in)
    x_out = Dense(2, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_in, outputs=x_out)


class MyModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


tests_clfdrift = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_clfdrift)


@pytest.fixture
def clfdrift_params(request):
    return tests_clfdrift[request.param]


@pytest.mark.parametrize('clfdrift_params', list(range(n_tests)), indirect=True)
def test_clfdrift(clfdrift_params):
    backend = clfdrift_params
    if backend.lower() == 'pytorch':
        model = MyModel(n_features)
    elif backend.lower() == 'tensorflow':
        model = mymodel((n_features,))
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
    else:
        assert cd is None
