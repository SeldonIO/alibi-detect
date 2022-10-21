from functools import partial
from itertools import product
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Callable, List
from alibi_detect.utils.frameworks import has_keops
from alibi_detect.utils.pytorch import GaussianRBF, mmd2_from_kernel_matrix
from alibi_detect.cd.pytorch.preprocess import HiddenOutput, preprocess_drift
if has_keops:
    from alibi_detect.cd.keops.mmd import MMDDriftKeops

n, n_hidden, n_classes = 500, 10, 5


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


n_features = [10]
n_enc = [None, 3]
preprocess = [
    (None, None),
    (preprocess_drift, {'model': HiddenOutput, 'layer': -1}),
    (preprocess_list, None)
]
update_x_ref = [{'last': 750}, {'reservoir_sampling': 750}, None]
preprocess_at_init = [True, False]
n_permutations = [10]
batch_size_permutations = [10, 1000000]
configure_kernel_from_x_ref = [True, False]
tests_mmddrift = list(product(n_features, n_enc, preprocess, n_permutations, preprocess_at_init, update_x_ref,
                              batch_size_permutations, configure_kernel_from_x_ref))
n_tests = len(tests_mmddrift)


@pytest.fixture
def mmd_params(request):
    return tests_mmddrift[request.param]


@pytest.mark.skipif(not has_keops, reason='Skipping since pykeops is not installed.')
@pytest.mark.parametrize('mmd_params', list(range(n_tests)), indirect=True)
def test_mmd(mmd_params):
    n_features, n_enc, preprocess, n_permutations, preprocess_at_init, update_x_ref, \
        batch_size_permutations, configure_kernel_from_x_ref = mmd_params

    np.random.seed(0)
    torch.manual_seed(0)

    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    preprocess_fn, preprocess_kwargs = preprocess
    to_list = False
    if hasattr(preprocess_fn, '__name__') and preprocess_fn.__name__ == 'preprocess_list':
        if not preprocess_at_init:
            return
        to_list = True
        x_ref = [_[None, :] for _ in x_ref]
    elif isinstance(preprocess_fn, Callable) and 'layer' in list(preprocess_kwargs.keys()) \
            and preprocess_kwargs['model'].__name__ == 'HiddenOutput':
        model = MyModel(n_features)
        layer = preprocess_kwargs['layer']
        preprocess_fn = partial(preprocess_fn, model=HiddenOutput(model=model, layer=layer))
    else:
        preprocess_fn = None

    cd = MMDDriftKeops(
        x_ref=x_ref,
        p_val=.05,
        preprocess_at_init=preprocess_at_init if isinstance(preprocess_fn, Callable) else False,
        update_x_ref=update_x_ref,
        preprocess_fn=preprocess_fn,
        configure_kernel_from_x_ref=configure_kernel_from_x_ref,
        n_permutations=n_permutations,
        batch_size_permutations=batch_size_permutations
    )
    x = x_ref.copy()
    preds = cd.predict(x, return_p_val=True)
    assert preds['data']['is_drift'] == 0 and preds['data']['p_val'] >= cd.p_val
    if isinstance(update_x_ref, dict):
        k = list(update_x_ref.keys())[0]
        assert cd.n == len(x) + len(x_ref)
        assert cd.x_ref.shape[0] == min(update_x_ref[k], len(x) + len(x_ref))

    x_h1 = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    if to_list:
        x_h1 = [_[None, :] for _ in x_h1]
    preds = cd.predict(x_h1, return_p_val=True)
    if preds['data']['is_drift'] == 1:
        assert preds['data']['p_val'] < preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] > preds['data']['distance_threshold']
    else:
        assert preds['data']['p_val'] >= preds['data']['threshold'] == cd.p_val
        assert preds['data']['distance'] <= preds['data']['distance_threshold']

    # ensure the keops MMD^2 estimate matches the pytorch implementation for the same kernel
    if not isinstance(x_ref, list) and update_x_ref is None:
        p_val, mmd2, distance_threshold = cd.score(x_h1)
        kernel = GaussianRBF(sigma=cd.kernel.sigma)
        if isinstance(preprocess_fn, Callable):
            x_ref, x_h1 = cd.preprocess(x_h1)
        x_ref = torch.from_numpy(x_ref).float()
        x_h1 = torch.from_numpy(x_h1).float()
        x_all = torch.cat([x_ref, x_h1], 0)
        kernel_mat = kernel(x_all, x_all)
        mmd2_torch = mmd2_from_kernel_matrix(kernel_mat, x_h1.shape[0])
        np.testing.assert_almost_equal(mmd2, mmd2_torch, decimal=6)
