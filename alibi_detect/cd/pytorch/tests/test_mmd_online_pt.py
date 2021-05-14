from functools import partial
from itertools import product
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Callable
from alibi_detect.cd.pytorch.mmd_online import MMDDriftOnlineTorch
from alibi_detect.cd.pytorch.preprocess import HiddenOutput, preprocess_drift

n, n_hidden, n_classes = 300, 10, 5


class MyModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


n_features = [10]
n_enc = [None, 3]
ert = [25]
window_size = [5]
preprocess = [
    (None, None),
    (preprocess_drift, {'model': HiddenOutput, 'layer': -1})
]
preprocess_x_ref = [True, False]
n_bootstraps = [200]
tests_mmddriftonline = list(product(n_features, n_enc, ert, window_size, preprocess,
                                    n_bootstraps, preprocess_x_ref))
n_tests = len(tests_mmddriftonline)


@pytest.fixture
def mmd_online_params(request):
    return tests_mmddriftonline[request.param]


@pytest.mark.parametrize('mmd_online_params', list(range(n_tests)), indirect=True)
def test_mmd_online(mmd_online_params):
    n_features, n_enc, ert, window_size, preprocess, n_bootstraps, preprocess_x_ref = mmd_online_params

    np.random.seed(0)
    torch.manual_seed(0)

    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    preprocess_fn, preprocess_kwargs = preprocess
    if isinstance(preprocess_fn, Callable) and 'layer' in list(preprocess_kwargs.keys()) \
            and preprocess_kwargs['model'].__name__ == 'HiddenOutput':
        model = MyModel(n_features)
        layer = preprocess_kwargs['layer']
        preprocess_fn = partial(preprocess_fn, model=HiddenOutput(model=model, layer=layer))
    else:
        preprocess_fn = None

    cd = MMDDriftOnlineTorch(
        x_ref=x_ref,
        ert=ert,
        window_size=window_size,
        preprocess_x_ref=preprocess_x_ref if isinstance(preprocess_fn, Callable) else False,
        preprocess_fn=preprocess_fn,
        n_bootstraps=n_bootstraps
    )

    x_h0 = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    detection_times_h0 = []
    test_stats_h0 = []
    for x_t in x_h0:
        pred_t = cd.predict(x_t, return_test_stat=True)
        test_stats_h0.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h0.append(pred_t['data']['time'])
            cd.reset()
    average_delay_h0 = (np.array(detection_times_h0) - window_size).mean()
    test_stats_h0 = [ts for ts in test_stats_h0 if ts is not None]

    assert ert/3 < average_delay_h0 < 3*ert
    assert min(detection_times_h0) >= window_size

    cd.reset()

    x_h1 = 1 + np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    detection_times_h1 = []
    test_stats_h1 = []
    for x_t in x_h1:
        pred_t = cd.predict(x_t, return_test_stat=True)
        test_stats_h1.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h1.append(pred_t['data']['time'])
            cd.reset()
    average_delay_h1 = (np.array(detection_times_h1) - window_size).mean()
    test_stats_h1 = [ts for ts in test_stats_h1 if ts is not None]

    assert np.abs(average_delay_h1) < ert/2
    assert min(detection_times_h0) >= window_size

    assert np.mean(test_stats_h1) > np.mean(test_stats_h0)
