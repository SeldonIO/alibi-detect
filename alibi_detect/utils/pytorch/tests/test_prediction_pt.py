import numpy as np
import pytest
import torch
import torch.nn as nn
from alibi_detect.utils.pytorch import predict_batch

n, n_features, n_classes, latent_dim = 100, 10, 5, 2
X = np.zeros((n, n_features), dtype=np.float32)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


model = MyModel()

AutoEncoder = nn.Sequential(
    nn.Linear(n_features, latent_dim),
    nn.Linear(latent_dim, n_features)
)

# model, batch size, dtype
tests_predict = [
    (model, 2, np.float32),
    (model, int(1e10), np.float32),
    (model, int(1e10), torch.float32),
    (AutoEncoder, 2, np.float32),
    (AutoEncoder, int(1e10), np.float32)
]
n_tests = len(tests_predict)


@pytest.fixture
def predict_batch_params(request):
    return tests_predict[request.param]


@pytest.mark.parametrize('predict_batch_params', list(range(n_tests)), indirect=True)
def test_predict_batch(predict_batch_params):
    model, batch_size, dtype = predict_batch_params
    preds = predict_batch(X, model, batch_size=batch_size, dtype=dtype)
    assert preds.dtype == dtype
    if isinstance(model, nn.Sequential):
        assert preds.shape == X.shape
    elif isinstance(model, nn.Module):
        assert preds.shape == (n, n_classes)
