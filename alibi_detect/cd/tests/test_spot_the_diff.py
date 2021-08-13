import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
from alibi_detect.cd import SpotTheDiffDrift
from alibi_detect.cd.pytorch.spot_the_diff import SpotTheDiffDriftTorch
from alibi_detect.cd.tensorflow.spot_the_diff import SpotTheDiffDriftTF

n, n_features = 100, 5


class MyKernelTF(tf.keras.Model):  # TODO: Support then test models using keras functional API
    def __init__(self, n_features: int):
        super().__init__()
        self.config = {'n_features': n_features}
        self.dense = Dense(20)

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.einsum('ji,ki->jk', self.dense(x), self.dense(y))

    def get_config(self) -> dict:
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MyKernelTorch(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


tests_stddrift = ['tensorflow', 'pytorch', 'PyToRcH', 'mxnet']
n_tests = len(tests_stddrift)


@pytest.fixture
def stddrift_params(request):
    return tests_stddrift[request.param]


@pytest.mark.parametrize('stddrift_params', list(range(n_tests)), indirect=True)
def test_stddrift(stddrift_params):
    backend = stddrift_params
    if backend.lower() == 'pytorch':
        kernel = MyKernelTorch(n_features)
    elif backend.lower() == 'tensorflow':
        kernel = MyKernelTF((n_features,))
    else:
        kernel = None
    x_ref = np.random.randn(*(n, n_features))

    try:
        cd = SpotTheDiffDrift(x_ref=x_ref, kernel=kernel, backend=backend)
    except NotImplementedError:
        cd = None

    if backend.lower() == 'pytorch':
        assert isinstance(cd._detector, SpotTheDiffDriftTorch)
    elif backend.lower() == 'tensorflow':
        assert isinstance(cd._detector, SpotTheDiffDriftTF)
    else:
        assert cd is None
