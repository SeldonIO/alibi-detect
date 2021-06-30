import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Tuple, Union
from alibi_detect.utils.pytorch import predict_batch

n, n_features, n_classes, latent_dim = 100, 10, 5, 2
x = np.zeros((n, n_features), dtype=np.float32)


class MyModel(nn.Module):
    def __init__(self, multi_out: bool = False):
        super(MyModel, self).__init__()
        self.dense = nn.Linear(n_features, n_classes)
        self.multi_out = multi_out

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = self.dense(x)
        if not self.multi_out:
            return out
        else:
            return out, out


AutoEncoder = nn.Sequential(
    nn.Linear(n_features, latent_dim),
    nn.Linear(latent_dim, n_features)
)


def id_fn(x: Union[np.ndarray, torch.Tensor, list]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x, list):
        return torch.from_numpy(np.concatenate(x, axis=0))
    else:
        return x


# model, batch size, dtype, preprocessing function, list as input
tests_predict = [
    (MyModel(multi_out=False), 2, np.float32, None, False),
    (MyModel(multi_out=False), int(1e10), np.float32, None, False),
    (MyModel(multi_out=False), int(1e10), torch.float32, None, False),
    (MyModel(multi_out=True), int(1e10), torch.float32, None, False),
    (MyModel(multi_out=False), int(1e10), np.float32, id_fn, False),
    (AutoEncoder, 2, np.float32, None, False),
    (AutoEncoder, int(1e10), np.float32, None, False),
    (AutoEncoder, int(1e10), torch.float32, None, False),
    (id_fn, 2, np.float32, None, False),
    (id_fn, 2, torch.float32, None, False),
    (id_fn, 2, np.float32, id_fn, True),
]
n_tests = len(tests_predict)


@pytest.fixture
def predict_batch_params(request):
    return tests_predict[request.param]


@pytest.mark.parametrize('predict_batch_params', list(range(n_tests)), indirect=True)
def test_predict_batch(predict_batch_params):
    model, batch_size, dtype, preprocess_fn, to_list = predict_batch_params
    x_batch = [x] if to_list else x
    preds = predict_batch(x_batch, model, batch_size=batch_size, preprocess_fn=preprocess_fn, dtype=dtype)
    if isinstance(preds, tuple):
        preds = preds[0]
    assert preds.dtype == dtype
    if isinstance(model, nn.Sequential) or hasattr(model, '__name__') and model.__name__ == 'id_fn':
        assert preds.shape == x.shape
    elif isinstance(model, nn.Module):
        assert preds.shape == (n, n_classes)
