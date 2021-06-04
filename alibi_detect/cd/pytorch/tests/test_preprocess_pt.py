from itertools import product
import numpy as np
import pytest
import torch
import torch.nn as nn
from alibi_detect.cd.pytorch import HiddenOutput

n, dim1, dim2, n_classes, latent_dim, n_hidden = 100, 2, 3, 5, 2, 7
n_features = dim1 * dim2
shape = (n, dim1, dim2)
X = np.random.rand(n * n_features).reshape(shape).astype('float32')


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.dense1 = nn.Linear(dim2, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        return self.dense2(x)


model2 = nn.Sequential(
    nn.Linear(dim2, n_hidden),
    nn.Linear(n_hidden, n_classes)
)

model = [1, 2]
layer = [0, 1, 2]
flatten = [True, False]
tests_hidden_output = list(product(model, layer, flatten))
n_tests_hidden_output = len(tests_hidden_output)


@pytest.fixture
def hidden_output_params(request):
    return tests_hidden_output[request.param]


@pytest.mark.parametrize('hidden_output_params', list(range(n_tests_hidden_output)), indirect=True)
def test_hidden_output(hidden_output_params):
    model, layer, flatten = hidden_output_params
    model = Model1() if model == 1 else model2
    X_hidden = HiddenOutput(model=model, layer=layer, flatten=flatten)(torch.from_numpy(X))
    if layer == 0:
        assert_shape = (n, dim1, dim2)
    elif layer == 1:
        assert_shape = (n, dim1, n_hidden)
    elif layer == 2:
        assert_shape = (n, dim1, n_classes)
    if flatten:
        assert_shape = (assert_shape[0],) + (np.prod(assert_shape[1:]),)
    assert X_hidden.shape == assert_shape
