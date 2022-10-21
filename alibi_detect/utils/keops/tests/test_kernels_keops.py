from itertools import product
import numpy as np
from alibi_detect.utils.frameworks import has_keops
import pytest
import torch
import torch.nn as nn
if has_keops:
    from pykeops.torch import LazyTensor
    from alibi_detect.utils.keops import DeepKernel, GaussianRBF

sigma = [None, np.array([1.]), np.array([1., 2.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
batch_size = [None, 5]
trainable = [True, False]
tests_gk = list(product(sigma, n_features, n_instances, batch_size, trainable))
n_tests_gk = len(tests_gk)


@pytest.fixture
def gaussian_kernel_params(request):
    return tests_gk[request.param]


@pytest.mark.skipif(not has_keops, reason='Skipping since pykeops is not installed.')
@pytest.mark.parametrize('gaussian_kernel_params', list(range(n_tests_gk)), indirect=True)
def test_gaussian_kernel(gaussian_kernel_params):
    sigma, n_features, n_instances, batch_size, trainable = gaussian_kernel_params

    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    if batch_size:
        xshape = (batch_size, ) + xshape
        yshape = (batch_size, ) + yshape
    sigma = sigma if sigma is None else torch.from_numpy(sigma).float()
    x = torch.from_numpy(np.random.random(xshape)).float()
    y = torch.from_numpy(np.random.random(yshape)).float()
    if batch_size:
        x_lazy, y_lazy = LazyTensor(x[:, :, None, :]), LazyTensor(y[:, None, :, :])
        x_lazy2 = LazyTensor(x[:, None, :, :])
    else:
        x_lazy, y_lazy = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
        x_lazy2 = LazyTensor(x[None, :, :])

    kernel = GaussianRBF(sigma=sigma, trainable=trainable)
    infer_sigma = True if sigma is None else False
    if trainable and infer_sigma:
        with pytest.raises(ValueError):
            kernel(x_lazy, y_lazy, infer_sigma=infer_sigma)
    else:
        k_xy = kernel(x_lazy, y_lazy, infer_sigma=infer_sigma)
        k_xx = kernel(x_lazy, x_lazy2, infer_sigma=infer_sigma)
        k_xy_shape = n_instances
        k_xx_shape = (n_instances[0], n_instances[0])
        axis = 1
        if batch_size:
            k_xy_shape = (batch_size, ) + k_xy_shape
            k_xx_shape = (batch_size, ) + k_xx_shape
            axis = 2
        assert k_xy.shape == k_xy_shape and k_xx.shape == k_xx_shape
        k_xx_argmax = k_xx.argmax(axis=axis)
        k_xx_min, k_xy_min = k_xx.min(axis=axis), k_xy.min(axis=axis)
        if batch_size:
            k_xx_argmax, k_xx_min, k_xy_min = k_xx_argmax[0], k_xx_min[0], k_xy_min[0]
        assert (torch.arange(n_instances[0]) == k_xx_argmax.cpu().view(-1)).all()
        assert (k_xx_min >= 0.).all() and (k_xy_min >= 0.).all()


if has_keops:
    class MyKernel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: LazyTensor, y: LazyTensor) -> LazyTensor:
            return (- ((x - y) ** 2).sum(-1)).exp()


n_features = [5]
n_instances = [(100, 100), (100, 75)]
kernel_a = ['GaussianRBF', 'MyKernel']
kernel_b = ['GaussianRBF', 'MyKernel', None]
eps = [0.5, 'trainable']
tests_dk = list(product(n_features, n_instances, kernel_a, kernel_b, eps))
n_tests_dk = len(tests_dk)


@pytest.fixture
def deep_kernel_params(request):
    return tests_dk[request.param]


@pytest.mark.skipif(not has_keops, reason='Skipping since pykeops is not installed.')
@pytest.mark.parametrize('deep_kernel_params', list(range(n_tests_dk)), indirect=True)
def test_deep_kernel(deep_kernel_params):
    n_features, n_instances, kernel_a, kernel_b, eps = deep_kernel_params

    proj = nn.Linear(n_features, n_features)
    kernel_a = MyKernel() if kernel_a == 'MyKernel' else GaussianRBF(trainable=True)
    if kernel_b == 'MyKernel':
        kernel_b = MyKernel()
    elif kernel_b == 'GaussianRBF':
        kernel_b = GaussianRBF(trainable=True)
    kernel = DeepKernel(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)

    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = torch.as_tensor(np.random.random(xshape).astype('float32'))
    y = torch.as_tensor(np.random.random(yshape).astype('float32'))
    x_proj, y_proj = kernel.proj(x), kernel.proj(y)
    x2_proj, x_proj = LazyTensor(x_proj[None, :, :]), LazyTensor(x_proj[:, None, :])
    y2_proj, y_proj = LazyTensor(y_proj[None, :, :]), LazyTensor(y_proj[:, None, :])
    if kernel_b:
        x2, x = LazyTensor(x[None, :, :]), LazyTensor(x[:, None, :])
        y2, y = LazyTensor(y[None, :, :]), LazyTensor(y[:, None, :])
    else:
        x, x2, y, y2 = None, None, None, None

    k_xy = kernel(x_proj, y2_proj, x, y2)
    k_yx = kernel(y_proj, x2_proj, y, x2)
    k_xx = kernel(x_proj, x2_proj, x, x2)
    assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
    assert (k_xx.Kmin_argKmin(1, axis=1)[0] > 0.).all()
    assert (torch.abs(k_xy.sum(1).sum(1) - k_yx.t().sum(1).sum(1)) < 1e-5).all()
