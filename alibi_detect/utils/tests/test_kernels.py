import dask.array as da
from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.kernels import gaussian_kernel

sigma = [np.array([1.]), np.array([1., 2.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
tests_gk = list(product(sigma, n_features, n_instances))
n_tests_gk = len(tests_gk)


@pytest.fixture
def gaussian_kernel_params(request):
    return tests_gk[request.param]


@pytest.mark.parametrize('gaussian_kernel_params', list(range(n_tests_gk)), indirect=True)
def test_gaussian_kernel(gaussian_kernel_params):
    sigma, n_features, n_instances = gaussian_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = np.random.random(xshape).astype('float32')
    y = np.random.random(yshape).astype('float32')
    xda = da.from_array(x, chunks=xshape)
    yda = da.from_array(y, chunks=yshape)

    gk_xy = gaussian_kernel(x, y, sigma=sigma)
    gk_xx = gaussian_kernel(x, x, sigma=sigma)

    gk_xy_da = gaussian_kernel(xda, yda, sigma=sigma).compute()
    gk_xx_da = gaussian_kernel(xda, xda, sigma=sigma).compute()

    assert gk_xy.shape == n_instances and gk_xx.shape == (xshape[0], xshape[0])
    assert (gk_xx == gk_xx_da).all() and (gk_xy == gk_xy_da).all()
    assert gk_xx.trace() == xshape[0] * len(sigma)
    assert (gk_xx > 0.).all() and (gk_xy > 0.).all()
