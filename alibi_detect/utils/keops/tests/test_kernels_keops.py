from itertools import product
import numpy as np
from alibi_detect.utils.frameworks import has_keops
import pytest
import torch
if has_keops:
    from pykeops.torch import LazyTensor
    from alibi_detect.utils.keops import GaussianRBF

sigma = [None, np.array([1.]), np.array([1., 2.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
tests_gk = list(product(sigma, n_features, n_instances, trainable))
n_tests_gk = len(tests_gk)


@pytest.fixture
def gaussian_kernel_params(request):
    return tests_gk[request.param]


@pytest.mark.skipif(not has_keops, reason='Skipping since pykeops is not installed.')
@pytest.mark.parametrize('gaussian_kernel_params', list(range(n_tests_gk)), indirect=True)
def test_gaussian_kernel(gaussian_kernel_params):
    sigma, n_features, n_instances, trainable = gaussian_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    sigma = sigma if sigma is None else torch.from_numpy(sigma).float()
    x = torch.from_numpy(np.random.random(xshape)).float()
    y = torch.from_numpy(np.random.random(yshape)).float()

    kernel = GaussianRBF(sigma=sigma, trainable=trainable)
    infer_sigma = True if sigma is None else False
    if trainable and infer_sigma:
        with pytest.raises(Exception):
            kernel(LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :]), infer_sigma=infer_sigma)
    else:
        k_xy = kernel(LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :]), infer_sigma=infer_sigma)
        k_xx = kernel(LazyTensor(x[:, None, :]), LazyTensor(x[None, :, :]), infer_sigma=infer_sigma)
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        assert (torch.arange(xshape[0]) == k_xx.argmax(axis=1).cpu().view(-1)).all()
        assert (k_xx.min(axis=1) >= 0.).all() and (k_xy.min(axis=1) >= 0.).all()
