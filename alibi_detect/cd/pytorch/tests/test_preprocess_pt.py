import numpy as np
import pytest
import torch
import torch.nn as nn
from alibi_detect.cd.pytorch import HiddenOutput

n, n_features, n_classes, latent_dim, n_hidden = 100, 10, 5, 2, 7
shape = (n_features,)
X = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.dense1 = nn.Linear(n_features, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        return self.dense2(x)


model2 = nn.Sequential(
    nn.Linear(n_features, n_hidden),
    nn.Linear(n_hidden, n_classes)
)

tests_hidden_output = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
n_tests_hidden_output = len(tests_hidden_output)


@pytest.fixture
def hidden_output_params(request):
    return tests_hidden_output[request.param]


@pytest.mark.parametrize('hidden_output_params', list(range(n_tests_hidden_output)), indirect=True)
def test_hidden_output(hidden_output_params):
    model, layer = hidden_output_params
    model = Model1() if model == 1 else model2
    X_hidden = HiddenOutput(model=model, layer=layer)(torch.from_numpy(X))
    if layer == 0:
        assert X_hidden.shape == (n, n_features)
    elif layer == 1:
        assert X_hidden.shape == (n, n_hidden)
    elif layer == 2:
        assert X_hidden.shape == (n, n_classes)
