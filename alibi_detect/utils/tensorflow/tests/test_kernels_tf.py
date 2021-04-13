from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from alibi_detect.utils.tensorflow import GaussianRBF

sigma = [None, np.array([1.]), np.array([1., 2.])]
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
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    kernel = GaussianRBF(sigma=sigma)
    infer_sigma = True if sigma is None else False
    k_xy = kernel(x, y, infer_sigma=infer_sigma).numpy()
    k_xx = kernel(x, x, infer_sigma=infer_sigma).numpy()

    assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
    np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
    assert (k_xx > 0.).all() and (k_xy > 0.).all()
