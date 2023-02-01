from itertools import product
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from alibi_detect.utils.frameworks import has_keops
from alibi_detect.utils.pytorch import GaussianRBF as GaussianRBFTorch
from alibi_detect.utils.pytorch import mmd2_from_kernel_matrix
if has_keops:
    from alibi_detect.cd.keops.learned_kernel import LearnedKernelDriftKeops
    from alibi_detect.utils.keops import GaussianRBF, BaseKernel, ProjKernel

n = 50  # number of instances used for the reference and test data samples in the tests


if has_keops:
    class MyKernel(BaseKernel):
        def __init__(self, n_features: int, proj: bool):
            super().__init__()
            sigma = .1
            self.kernel_a = GaussianRBF(trainable=True, sigma=torch.Tensor([sigma]))
            self.log_sigma_a = self.kernel_a.parameter_dict['log-sigma'].value
            self.has_proj = proj
            if proj:
                self.proj = nn.Linear(n_features, 2)
                self.kernel_b = GaussianRBF(trainable=True, sigma=torch.Tensor([sigma]))
                self.proj_kernel = ProjKernel(self.proj, self.kernel_b)
                self.comp_kernel = self.proj_kernel + self.kernel_a
                self.log_sigma_b = self.kernel_b.parameter_dict['log-sigma'].value
            else:
                self.comp_kernel = self.kernel_a

        def kernel_function(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            infer_parameter: Optional[bool] = False
        ) -> torch.Tensor:
            return self.comp_kernel(x, y, infer_parameter)


# test List[Any] inputs to the detector
def identity_fn(x: Union[torch.Tensor, list]) -> torch.Tensor:
    if isinstance(x, list):
        return torch.from_numpy(np.array(x))
    else:
        return x


p_val = [.05]
n_features = [4]
preprocess_at_init = [True, False]
update_x_ref = [None, {'reservoir_sampling': 1000}]
preprocess_fn = [None, identity_fn]
n_permutations = [10]
batch_size_permutations = [5, 1000000]
train_size = [.5]
retrain_from_scratch = [True]
batch_size_predict = [1000000]
preprocess_batch = [None, identity_fn]
has_proj = [True, False]
tests_lkdrift = list(product(p_val, n_features, preprocess_at_init, update_x_ref, preprocess_fn,
                             n_permutations, batch_size_permutations, train_size, retrain_from_scratch,
                             batch_size_predict, preprocess_batch, has_proj))
n_tests = len(tests_lkdrift)


@pytest.fixture
def lkdrift_params(request):
    return tests_lkdrift[request.param]


@pytest.mark.skipif(not has_keops, reason='Skipping since pykeops is not installed.')
@pytest.mark.parametrize('lkdrift_params', list(range(n_tests)), indirect=True)
def test_lkdrift(lkdrift_params):
    p_val, n_features, preprocess_at_init, update_x_ref, preprocess_fn, \
        n_permutations, batch_size_permutations, train_size, retrain_from_scratch, \
        batch_size_predict, preprocess_batch, has_proj = lkdrift_params

    np.random.seed(0)
    torch.manual_seed(0)

    kernel = MyKernel(n_features, has_proj)
    x_ref = np.random.randn(*(n, n_features)).astype(np.float32)
    x_test1 = np.ones_like(x_ref)
    to_list = False
    if preprocess_batch is not None and preprocess_fn is None:
        to_list = True
        x_ref = [_ for _ in x_ref]
        update_x_ref = None

    cd = LearnedKernelDriftKeops(
        x_ref=x_ref,
        kernel=kernel,
        p_val=p_val,
        preprocess_at_init=preprocess_at_init,
        update_x_ref=update_x_ref,
        preprocess_fn=preprocess_fn,
        n_permutations=n_permutations,
        batch_size_permutations=batch_size_permutations,
        train_size=train_size,
        retrain_from_scratch=retrain_from_scratch,
        batch_size_predict=batch_size_predict,
        preprocess_batch_fn=preprocess_batch,
        batch_size=32,
        epochs=1
    )

    x_test0 = x_ref.copy()
    preds_0 = cd.predict(x_test0)
    assert cd.n == len(x_test0) + len(x_ref)
    assert preds_0['data']['is_drift'] == 0

    if to_list:
        x_test1 = [_ for _ in x_test1]
    preds_1 = cd.predict(x_test1)
    assert cd.n == len(x_test1) + len(x_test0) + len(x_ref)
    assert preds_1['data']['is_drift'] == 1
    assert preds_0['data']['distance'] < preds_1['data']['distance']

    # ensure the keops MMD^2 estimate matches the pytorch implementation for the same kernel
    if not isinstance(x_ref, list) and update_x_ref is None and not has_proj:
        if isinstance(preprocess_fn, Callable):
            x_ref, x_test1 = cd.preprocess(x_test1)
        n_ref, n_test = x_ref.shape[0], x_test1.shape[0]
        x_all = torch.from_numpy(np.concatenate([x_ref, x_test1], axis=0)).float()
        perms = [torch.randperm(n_ref + n_test) for _ in range(n_permutations)]
        mmd2 = cd._mmd2(x_all, perms, n_ref, n_test)[0]

        if isinstance(preprocess_batch, Callable):
            x_all = preprocess_batch(x_all)
        kernel = GaussianRBFTorch(sigma=cd.kernel.kernel_a.sigma.cpu())
        kernel_mat = kernel(x_all, x_all)
        mmd2_torch = mmd2_from_kernel_matrix(kernel_mat, n_test)
        np.testing.assert_almost_equal(mmd2.cpu(), mmd2_torch.cpu(), decimal=6)
