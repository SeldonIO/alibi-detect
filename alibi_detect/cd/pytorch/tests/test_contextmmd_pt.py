from functools import partial
from itertools import product
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Callable, List
from alibi_detect.cd.pytorch.context_aware import ContextMMDDriftTorch
from alibi_detect.cd.pytorch.preprocess import HiddenOutput, preprocess_drift


class MyModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


# test List[Any] inputs to the detector
def preprocess_list(x: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(x, axis=0)


n = 250
n_features = [10]
n_enc = [None, 3]
preprocess = [
    (None, None),
    (preprocess_drift, {'model': HiddenOutput, 'layer': -1}),
    (preprocess_list, None)
]
update_ref = [{'last': 750}, None]
preprocess_x_ref = [True, False]
n_permutations = [10]
tests_context_mmddrift = list(product(n_features, n_enc, preprocess,
                              n_permutations, update_ref, preprocess_x_ref))
n_tests = len(tests_context_mmddrift)


@pytest.fixture
def context_mmd_params(request):
    return tests_context_mmddrift[request.param]


@pytest.mark.parametrize('context_mmd_params', list(range(n_tests)), indirect=True)
def test_context_mmd(context_mmd_params):
    n_features, n_enc, preprocess, n_permutations, update_ref, preprocess_x_ref = context_mmd_params

    np.random.seed(0)
    torch.manual_seed(0)

    c_ref = np.random.randn(*(n, 1)).astype(np.float32)
    x_ref = c_ref + np.random.randn(*(n, n_features)).astype(np.float32)
    preprocess_fn, preprocess_kwargs = preprocess
    to_list = False
    if hasattr(preprocess_fn, '__name__') and preprocess_fn.__name__ == 'preprocess_list':
        if not preprocess_x_ref:
            pytest.skip("Skip tests where preprocess_x_ref=False and x_ref is list.")
        to_list = True
        x_ref = [_[None, :] for _ in x_ref]
    elif isinstance(preprocess_fn, Callable) and 'layer' in list(preprocess_kwargs.keys()) \
            and preprocess_kwargs['model'].__name__ == 'HiddenOutput':
        model = MyModel(n_features)
        layer = preprocess_kwargs['layer']
        preprocess_fn = partial(preprocess_fn, model=HiddenOutput(model=model, layer=layer))
    else:
        preprocess_fn = None

    cd = ContextMMDDriftTorch(
        x_ref=x_ref,
        c_ref=c_ref,
        p_val=.05,
        preprocess_x_ref=preprocess_x_ref if isinstance(preprocess_fn, Callable) else False,
        update_ref=update_ref,
        preprocess_fn=preprocess_fn,
        n_permutations=n_permutations
    )
    c = c_ref.copy()
    x = x_ref.copy()
    preds = cd.predict(x, c, return_p_val=True, return_distance=False, return_coupling=True)
    assert preds['data']['is_drift'] == 0 and preds['data']['p_val'] >= cd.p_val
    assert preds['data']['distance'] is None
    assert isinstance(preds['data']['coupling_xy'], np.ndarray)

    if isinstance(update_ref, dict):
        k = list(update_ref.keys())[0]
        assert cd.n == len(x) + len(x_ref)
        assert cd.x_ref.shape[0] == min(update_ref[k], len(x) + len(x_ref))
        assert cd.c_ref.shape[0] == min(update_ref[k], len(x) + len(c_ref))

    c_h1 = np.random.randn(*(n, 1)).astype(np.float32)
    x_h1 = c_h1 + np.random.randn(*(n, n_features)).astype(np.float32)
    if to_list:
        x_h1 = [_[None, :] for _ in x_h1]
    preds = cd.predict(x_h1, c_h1, return_p_val=True, return_distance=True, return_coupling=False)
    if preds['data']['is_drift'] == 1:
        assert preds['data']['p_val'] < preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] > preds['data']['distance_threshold']
    else:
        assert preds['data']['p_val'] >= preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] <= preds['data']['distance_threshold']
    assert 'coupling_xy' not in preds['data']
