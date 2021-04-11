from itertools import product
import numpy as np
import pytest
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from alibi_detect.cd.pytorch.classifier import ClassifierDriftTorch
from alibi_detect.utils.metrics import accuracy

n = 100


class MyModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


def f1_adj(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, np.round(y_pred))


threshold = [.6]
n_features = [4]
metric_fns = [accuracy, f1_adj]
n_folds = [None, 2]
train_size = [.5]
update_x_ref = [None, {'last': 1000}, {'reservoir_sampling': 1000}]
tests_clfdrift = list(product(threshold, n_features, metric_fns, n_folds,
                              train_size, update_x_ref))
n_tests = len(tests_clfdrift)


@pytest.fixture
def clfdrift_params(request):
    return tests_clfdrift[request.param]


@pytest.mark.parametrize('clfdrift_params', list(range(n_tests)), indirect=True)
def test_clfdrift(clfdrift_params):
    threshold, n_features, metric_fn, n_folds, train_size, update_x_ref = clfdrift_params

    np.random.seed(0)
    torch.manual_seed(0)

    model = MyModel(n_features)
    x_ref = np.random.randn(*(n, n_features)).astype(np.float32)
    x_test0 = x_ref.copy()
    x_test1 = np.ones_like(x_ref)

    cd = ClassifierDriftTorch(
        x_ref=x_ref,
        model=model,
        threshold=threshold,
        update_x_ref=update_x_ref,
        train_size=train_size,
        n_folds=n_folds,
        metric_fn=metric_fn,
        batch_size=1
    )

    preds_0 = cd.predict(x_test0)
    assert preds_0['data'][f'{metric_fn.__name__}'] <= threshold
    assert cd.n == x_test0.shape[0] + x_ref.shape[0]
    assert preds_0['data']['is_drift'] == 0

    preds_1 = cd.predict(x_test1)
    assert preds_1['data'][f'{metric_fn.__name__}'] > threshold
    assert cd.n == x_test1.shape[0] + x_test0.shape[0] + x_ref.shape[0]
    assert preds_1['data']['is_drift'] == 1

    assert cd.meta['params']['metric_fn'] == metric_fn.__name__
