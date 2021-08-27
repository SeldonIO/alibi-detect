from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from alibi_detect.utils.tensorflow import GaussianRBF, DeepKernel

sigma = [None, np.array([1.]), np.array([1., 2.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
tests_gk = list(product(sigma, n_features, n_instances, trainable))
n_tests_gk = len(tests_gk)


@pytest.fixture
def gaussian_kernel_params(request):
    return tests_gk[request.param]


@pytest.mark.parametrize('gaussian_kernel_params', list(range(n_tests_gk)), indirect=True)
def test_gaussian_kernel(gaussian_kernel_params):
    sigma, n_features, n_instances, trainable = gaussian_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    kernel = GaussianRBF(sigma=sigma, trainable=trainable)
    infer_sigma = True if sigma is None else False
    if trainable and infer_sigma:
        with pytest.raises(Exception):
            kernel(x, y, infer_sigma=infer_sigma)
    else:
        k_xy = kernel(x, y, infer_sigma=infer_sigma).numpy()
        k_xx = kernel(x, x, infer_sigma=infer_sigma).numpy()
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
        assert (k_xx > 0.).all() and (k_xy > 0.).all()


class MyKernel(tf.keras.Model):  # TODO: Support then test models using keras functional API
    def __init__(self, n_features: int):
        super().__init__()
        self.dense = Dense(20)

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.einsum('ji,ki->jk', self.dense(x), self.dense(y))


n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
kernel_a = [GaussianRBF(trainable=True), MyKernel]
kernel_b = [GaussianRBF(trainable=True), MyKernel, None]
eps = [0.5, 'trainable']
tests_dk = list(product(n_features, n_instances, kernel_a, kernel_b, eps))
n_tests_dk = len(tests_dk)


@pytest.fixture
def deep_kernel_params(request):
    return tests_dk[request.param]


@pytest.mark.parametrize('deep_kernel_params', list(range(n_tests_dk)), indirect=True)
def test_deep_kernel(deep_kernel_params):
    n_features, n_instances, kernel_a, kernel_b, eps = deep_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    proj = tf.keras.Sequential([Input(shape=(n_features,)), Dense(n_features)])
    kernel_a = kernel_a(n_features) if kernel_a == MyKernel else kernel_a
    kernel_b = kernel_b(n_features) if kernel_b == MyKernel else kernel_b

    kernel = DeepKernel(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)

    k_xy = kernel(x, y).numpy()
    k_yx = kernel(y, x).numpy()
    k_xx = kernel(x, x).numpy()
    assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
    assert (np.diag(k_xx) > 0.).all()
    np.testing.assert_almost_equal(k_xy, np.transpose(k_yx), decimal=5)
